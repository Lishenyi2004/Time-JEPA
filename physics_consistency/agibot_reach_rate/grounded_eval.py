#!/usr/bin/env python3
"""GT-grounded evaluation for the AgiBot current arm reach-rate law."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from physics_consistency.agibot.urdf_fk import (
    G2ForwardKinematics,
    apply_rigid_alignment,
    load_calibration,
)
from physics_consistency.agibot_reach_rate.common import (
    ARMS,
    P_SLICES,
    PROTOCOL,
    Q_INDICES,
    TARGET_DEFINITION,
    TARGET_NAMES,
    V_INDICES,
    atomic_json_dump,
    json_safe,
    sha256_file,
)
from physics_consistency.agibot_reach_rate.metrics import (
    direction_angle_metrics,
    macro_average,
    paired_preference,
    regression_metrics,
)


FORMULAS = (
    "pred_q_pred_v",
    "gt_q_pred_v",
    "pred_q_gt_v",
    "gt_q_gt_v",
    "raw_geometry_gt_v",
    "zero",
    "train_mean",
    "wrong_arm_q",
    "time_shift_q",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _unit_direction(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector, axis=-1)
    valid = np.all(np.isfinite(vector), axis=-1) & (norm > 1e-8)
    result = np.zeros_like(vector)
    result[valid] = vector[valid] / norm[valid, None]
    return result, valid


def _aligned_fk(
    fk: G2ForwardKinematics,
    q: np.ndarray,
    arm: str,
    calibration: dict,
) -> np.ndarray:
    return apply_rigid_alignment(fk.forward(q, arm), calibration)


def _scalar(payload: dict, key: str, default=None):
    if key not in payload:
        return default
    value = np.asarray(payload[key])
    return value.item() if value.ndim == 0 else value


def load_prediction_file(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    names = tuple(str(item) for item in payload.get("target_names", ()).tolist())
    if names != TARGET_NAMES:
        raise ValueError(
            f"Prediction target schema mismatch: {names} != {TARGET_NAMES}"
        )
    if _scalar(payload, "protocol", "") != PROTOCOL:
        raise ValueError("Prediction artifact uses the wrong reach-rate protocol")
    if _scalar(payload, "target_definition", "") != TARGET_DEFINITION:
        raise ValueError("Prediction artifact uses the wrong target definition")

    shape = payload["prediction"].shape
    if len(shape) != 3 or shape[-1] != len(TARGET_NAMES):
        raise ValueError(f"Expected prediction [N,T,20], got {shape}")
    for key in ("target", "valid"):
        if payload[key].shape != shape:
            raise ValueError(f"{key} shape mismatch: {payload[key].shape} != {shape}")
    expected = {
        "endpoint_position": (*shape[:2], 6),
        "endpoint_valid": (*shape[:2], 2),
        "reach_rate_gt": (*shape[:2], 2),
        "reach_rate_valid": (*shape[:2], 2),
        "target_sample_times": shape[:2],
    }
    for key, expected_shape in expected.items():
        if key not in payload or payload[key].shape != expected_shape:
            actual = None if key not in payload else payload[key].shape
            raise ValueError(f"{key} shape mismatch: {actual} != {expected_shape}")
    for key in (
        "video_id",
        "episode_key",
        "window_key",
        "task_id",
        "view_key",
    ):
        if key not in payload or len(payload[key]) != shape[0]:
            raise ValueError(f"Prediction artifact needs row metadata {key!r}")
    if payload["target_mean"].shape != (len(TARGET_NAMES),):
        raise ValueError("target_mean must have shape [20]")
    if payload["target_std"].shape != (len(TARGET_NAMES),):
        raise ValueError("target_std must have shape [20]")
    payload["path"] = str(path)
    return payload


def load_gate(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing reach-rate GT Gate: {path}")
    gate = json.loads(path.read_text())
    if gate.get("status") != "PASS":
        raise RuntimeError(f"Reach-rate GT Gate is not PASS: {path}")
    if gate.get("protocol") != PROTOCOL:
        raise RuntimeError("GT Gate protocol mismatch")
    if gate.get("target_definition") != TARGET_DEFINITION:
        raise RuntimeError("GT Gate target-definition mismatch")
    gate["_path"] = str(path)
    gate["_sha256"] = sha256_file(path)
    return gate


def _shift_within_window(
    values: np.ndarray,
    valid: np.ndarray,
    shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use q(t-shift) at t without leaking across independent windows."""

    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool)
    if shift <= 0 or shift >= values.shape[1]:
        raise ValueError(
            f"time_shift_tokens must be in [1,{values.shape[1] - 1}]"
        )
    shifted = np.zeros_like(values)
    shifted_valid = np.zeros_like(valid)
    shifted[:, shift:] = values[:, :-shift]
    shifted_valid[:, shift:] = valid[:, :-shift]
    return shifted, shifted_valid


def _arm_state(
    payload: dict,
    gate: dict,
    fk: G2ForwardKinematics,
    calibrations: dict,
    arm: str,
    time_shift_tokens: int,
) -> dict:
    arm_index = ARMS.index(arm)
    other = ARMS[1 - arm_index]
    prediction = np.asarray(payload["prediction"], dtype=np.float64)
    target = np.asarray(payload["target"], dtype=np.float64)
    label_valid = np.asarray(payload["valid"]) > 0

    q_pred = prediction[..., Q_INDICES[arm]]
    q_gt = target[..., Q_INDICES[arm]]
    q_component_valid = label_valid[..., Q_INDICES[arm]]
    q_valid = np.all(q_component_valid, axis=-1)
    other_q_pred = prediction[..., Q_INDICES[other]]
    other_q_valid = np.all(label_valid[..., Q_INDICES[other]], axis=-1)
    v_pred = prediction[..., V_INDICES[arm]]
    v_gt = target[..., V_INDICES[arm]]
    v_component_valid = label_valid[..., V_INDICES[arm]]
    v_valid = np.all(v_component_valid, axis=-1)
    endpoint = np.asarray(
        payload["endpoint_position"][..., P_SLICES[arm]], dtype=np.float64
    )
    endpoint_valid = np.asarray(
        payload["endpoint_valid"][..., arm_index]
    ) > 0
    reach_gt = np.asarray(
        payload["reach_rate_gt"][..., arm_index], dtype=np.float64
    )
    reach_valid = np.asarray(
        payload["reach_rate_valid"][..., arm_index]
    ) > 0

    shoulder = np.asarray(
        gate["shoulder_points_dataset"][arm], dtype=np.float64
    )
    pred_endpoint = _aligned_fk(
        fk, q_pred, arm, calibrations[arm]
    )
    gt_endpoint = _aligned_fk(
        fk, q_gt, arm, calibrations[arm]
    )
    wrong_endpoint = _aligned_fk(
        fk, other_q_pred, arm, calibrations[arm]
    )
    shifted_q, shifted_q_valid = _shift_within_window(
        q_pred, q_valid, int(time_shift_tokens)
    )
    shifted_endpoint = _aligned_fk(
        fk, shifted_q, arm, calibrations[arm]
    )

    pred_direction, pred_direction_valid = _unit_direction(
        pred_endpoint - shoulder
    )
    gt_direction, gt_direction_valid = _unit_direction(
        gt_endpoint - shoulder
    )
    raw_direction, raw_direction_valid = _unit_direction(endpoint - shoulder)
    wrong_direction, wrong_direction_valid = _unit_direction(
        wrong_endpoint - shoulder
    )
    shifted_direction, shifted_direction_valid = _unit_direction(
        shifted_endpoint - shoulder
    )

    base_valid = reach_valid
    formula_values = {
        "pred_q_pred_v": np.sum(pred_direction * v_pred, axis=-1),
        "gt_q_pred_v": np.sum(gt_direction * v_pred, axis=-1),
        "pred_q_gt_v": np.sum(pred_direction * v_gt, axis=-1),
        "gt_q_gt_v": np.sum(gt_direction * v_gt, axis=-1),
        "raw_geometry_gt_v": np.sum(raw_direction * v_gt, axis=-1),
        "zero": np.zeros_like(reach_gt),
        "train_mean": np.full_like(
            reach_gt,
            float(
                gate["arms"][arm]["reach_rate_train_stats"]["mean_mps"]
            ),
        ),
        "wrong_arm_q": np.sum(wrong_direction * v_pred, axis=-1),
        "time_shift_q": np.sum(shifted_direction * v_pred, axis=-1),
    }
    formula_valid = {
        "pred_q_pred_v": (
            base_valid & q_valid & v_valid & pred_direction_valid
        ),
        "gt_q_pred_v": (
            base_valid & q_valid & v_valid & gt_direction_valid
        ),
        "pred_q_gt_v": (
            base_valid & q_valid & v_valid & pred_direction_valid
        ),
        "gt_q_gt_v": (
            base_valid & q_valid & v_valid & gt_direction_valid
        ),
        "raw_geometry_gt_v": (
            base_valid
            & endpoint_valid
            & v_valid
            & raw_direction_valid
        ),
        # Baselines use exactly the formal formula's support so their error is
        # directly comparable rather than benefiting from a different mask.
        "zero": base_valid & q_valid & v_valid & pred_direction_valid,
        "train_mean": base_valid & q_valid & v_valid & pred_direction_valid,
        "wrong_arm_q": (
            base_valid
            & other_q_valid
            & v_valid
            & wrong_direction_valid
        ),
        "time_shift_q": (
            base_valid
            & shifted_q_valid
            & v_valid
            & shifted_direction_valid
        ),
    }
    threshold = float(
        gate["arms"][arm]["reach_rate_train_stats"][
            "active_abs_q75_mps"
        ]
    )
    return {
        "q_pred": q_pred,
        "q_gt": q_gt,
        "q_component_valid": q_component_valid,
        "q_valid": q_valid,
        "v_pred": v_pred,
        "v_gt": v_gt,
        "v_component_valid": v_component_valid,
        "v_valid": v_valid,
        "raw_direction": raw_direction,
        "pred_direction": pred_direction,
        "direction_valid": (
            q_valid
            & endpoint_valid
            & pred_direction_valid
            & raw_direction_valid
        ),
        "reach_gt": reach_gt,
        "reach_valid": reach_valid,
        "active": reach_valid & (np.abs(reach_gt) > threshold),
        "active_threshold_mps": threshold,
        "formula_values": formula_values,
        "formula_valid": formula_valid,
    }


def _point_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    train_mean: float,
    train_std: float,
) -> dict:
    # Point metrics use train std for both normalized summaries.  The formula
    # itself uses the exact frozen mean-absolute reach-rate denominator.
    result = regression_metrics(
        prediction,
        target,
        valid,
        train_mean,
        train_std,
        train_std,
    )
    if "nmae_train_mean_abs" in result:
        result["nmae_train_std"] = result.pop("nmae_train_mean_abs")
    return result


def _summarize_arm(
    payload: dict,
    gate: dict,
    state: dict,
    arm: str,
    row_mask: np.ndarray,
) -> dict:
    row_mask = np.asarray(row_mask, dtype=bool)
    mean = np.asarray(payload["target_mean"], dtype=np.float64)
    std = np.asarray(payload["target_std"], dtype=np.float64)
    q_components = []
    for index, target_index in enumerate(Q_INDICES[arm]):
        q_components.append(
            {
                "name": f"q_{arm}_{index}",
                **_point_metrics(
                    state["q_pred"][row_mask, :, index],
                    state["q_gt"][row_mask, :, index],
                    state["q_component_valid"][row_mask, :, index],
                    float(mean[target_index]),
                    float(std[target_index]),
                ),
            }
        )
    v_components = []
    for index, target_index in enumerate(V_INDICES[arm]):
        v_components.append(
            {
                "name": TARGET_NAMES[target_index],
                **_point_metrics(
                    state["v_pred"][row_mask, :, index],
                    state["v_gt"][row_mask, :, index],
                    state["v_component_valid"][row_mask, :, index],
                    float(mean[target_index]),
                    float(std[target_index]),
                ),
            }
        )

    train_stats = gate["arms"][arm]["reach_rate_train_stats"]
    formula_metrics = {}
    for name in FORMULAS:
        valid = state["formula_valid"][name] & row_mask[:, None]
        active = valid & state["active"]
        formula_metrics[name] = {
            "all": regression_metrics(
                state["formula_values"][name],
                state["reach_gt"],
                valid,
                float(train_stats["mean_mps"]),
                float(train_stats["std_mps"]),
                float(train_stats["mean_abs_mps"]),
            ),
            "active": regression_metrics(
                state["formula_values"][name],
                state["reach_gt"],
                active,
                float(train_stats["mean_mps"]),
                float(train_stats["std_mps"]),
                float(train_stats["mean_abs_mps"]),
            ),
        }

    formal_valid = (
        state["formula_valid"]["pred_q_pred_v"]
        & row_mask[:, None]
    )
    wrong_valid = (
        formal_valid & state["formula_valid"]["wrong_arm_q"]
    )
    shift_valid = (
        formal_valid & state["formula_valid"]["time_shift_q"]
    )
    controls = {
        "wrong_arm_preference": paired_preference(
            state["formula_values"]["pred_q_pred_v"],
            state["formula_values"]["wrong_arm_q"],
            state["reach_gt"],
            wrong_valid,
        ),
        "time_shift_preference": paired_preference(
            state["formula_values"]["pred_q_pred_v"],
            state["formula_values"]["time_shift_q"],
            state["reach_gt"],
            shift_valid,
        ),
        "active_wrong_arm_preference": paired_preference(
            state["formula_values"]["pred_q_pred_v"],
            state["formula_values"]["wrong_arm_q"],
            state["reach_gt"],
            wrong_valid & state["active"],
        ),
        "active_time_shift_preference": paired_preference(
            state["formula_values"]["pred_q_pred_v"],
            state["formula_values"]["time_shift_q"],
            state["reach_gt"],
            shift_valid & state["active"],
        ),
    }
    return {
        "arm": arm,
        "active_threshold_mps": state["active_threshold_mps"],
        "q_point": {
            "components": q_components,
            "macro": macro_average(
                q_components,
                ("mae", "rmse", "nrmse_train_std", "pearson", "r2"),
            ),
        },
        "velocity_point": {
            "components": v_components,
            "macro": macro_average(
                v_components,
                ("mae", "rmse", "nrmse_train_std", "pearson", "r2"),
            ),
        },
        "direction_to_raw_endpoint": direction_angle_metrics(
            state["pred_direction"][row_mask],
            state["raw_direction"][row_mask],
            state["direction_valid"][row_mask],
        ),
        "formulas": formula_metrics,
        "controls": controls,
    }


def _flat_summary(summary: dict) -> dict:
    formulas = summary["formulas"]
    controls = summary["controls"]
    active_formula_mae = formulas["pred_q_pred_v"]["active"].get("mae")
    active_zero_mae = formulas["zero"]["active"].get("mae")
    active_train_mean_mae = formulas["train_mean"]["active"].get("mae")
    return {
        "q_macro_nrmse": summary["q_point"]["macro"].get(
            "nrmse_train_std"
        ),
        "direction_mean_deg": summary["direction_to_raw_endpoint"].get(
            "mean_deg"
        ),
        "velocity_macro_nrmse": summary["velocity_point"]["macro"].get(
            "nrmse_train_std"
        ),
        "formula_mae": formulas["pred_q_pred_v"]["all"].get("mae"),
        "formula_nmae": formulas["pred_q_pred_v"]["all"].get(
            "nmae_train_mean_abs"
        ),
        "formula_rmse": formulas["pred_q_pred_v"]["all"].get("rmse"),
        "formula_pearson": formulas["pred_q_pred_v"]["all"].get(
            "pearson"
        ),
        "active_formula_mae": active_formula_mae,
        "active_formula_nmae": formulas["pred_q_pred_v"]["active"].get(
            "nmae_train_mean_abs"
        ),
        "active_formula_rmse": formulas["pred_q_pred_v"]["active"].get(
            "rmse"
        ),
        "active_formula_pearson": formulas["pred_q_pred_v"]["active"].get(
            "pearson"
        ),
        "gt_q_pred_v_mae": formulas["gt_q_pred_v"]["all"].get("mae"),
        "pred_q_gt_v_mae": formulas["pred_q_gt_v"]["all"].get("mae"),
        "gt_q_gt_v_mae": formulas["gt_q_gt_v"]["all"].get("mae"),
        "raw_geometry_oracle_mae": formulas["raw_geometry_gt_v"][
            "all"
        ].get("mae"),
        "zero_mae": formulas["zero"]["all"].get("mae"),
        "train_mean_mae": formulas["train_mean"]["all"].get("mae"),
        "active_zero_mae": active_zero_mae,
        "active_train_mean_mae": active_train_mean_mae,
        "active_error_reduction_vs_zero": (
            active_zero_mae - active_formula_mae
            if active_zero_mae is not None
            and active_formula_mae is not None
            else None
        ),
        "active_error_reduction_vs_train_mean": (
            active_train_mean_mae - active_formula_mae
            if active_train_mean_mae is not None
            and active_formula_mae is not None
            else None
        ),
        "wrong_arm_preference": controls["wrong_arm_preference"].get(
            "tie_aware_accuracy"
        ),
        "time_shift_preference": controls["time_shift_preference"].get(
            "tie_aware_accuracy"
        ),
        "active_wrong_arm_preference": controls[
            "active_wrong_arm_preference"
        ].get("tie_aware_accuracy"),
        "active_time_shift_preference": controls[
            "active_time_shift_preference"
        ].get("tie_aware_accuracy"),
        "valid_tokens": formulas["pred_q_pred_v"]["all"].get("count", 0),
        "active_valid_tokens": formulas["pred_q_pred_v"]["active"].get(
            "count", 0
        ),
    }


def _macro_flat(left: dict, right: dict) -> dict:
    output = {}
    for key in left:
        values = np.asarray([left[key], right[key]], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[key] = float(np.mean(finite)) if len(finite) else np.nan
    return output


def _table_row(
    identity: dict,
    arm: str,
    summary: dict,
    extra: dict,
) -> dict:
    return {
        **identity,
        **extra,
        "arm": arm,
        **_flat_summary(summary),
    }


def evaluate_prediction_file(
    prediction_path: str | Path,
    gate_path: str | Path,
    output_dir: str | Path,
    split_name: str | None = None,
    time_shift_tokens: int = 1,
) -> dict:
    payload = load_prediction_file(prediction_path)
    gate = load_gate(gate_path)
    artifact_split = str(_scalar(payload, "split_name", ""))
    split_name = split_name or artifact_split
    if split_name != artifact_split:
        raise ValueError(
            f"Requested split {split_name!r} != artifact {artifact_split!r}"
        )

    urdf_path = Path(gate["urdf"]["path"]).expanduser().resolve()
    if sha256_file(urdf_path) != gate["urdf"]["sha256"]:
        raise RuntimeError("URDF content changed after the passed GT Gate")
    calibration_path = Path(gate["calibration_path"]).expanduser().resolve()
    calibrations = load_calibration(calibration_path)
    fk = G2ForwardKinematics(urdf_path)
    states = {
        arm: _arm_state(
            payload,
            gate,
            fk,
            calibrations,
            arm,
            time_shift_tokens,
        )
        for arm in ARMS
    }
    all_rows = np.ones(payload["prediction"].shape[0], dtype=bool)
    arm_summaries = {
        arm: _summarize_arm(
            payload, gate, states[arm], arm, all_rows
        )
        for arm in ARMS
    }
    arm_flat = {arm: _flat_summary(arm_summaries[arm]) for arm in ARMS}

    identity = {
        "model": str(_scalar(payload, "model_name", "")),
        "seed": int(_scalar(payload, "seed", -1)),
        "split": split_name,
    }
    episode_rows = []
    for episode in sorted(set(payload["episode_key"].astype(str).tolist())):
        mask = payload["episode_key"].astype(str) == episode
        tasks = set(np.asarray(payload["task_id"])[mask].astype(int).tolist())
        if len(tasks) != 1:
            raise RuntimeError(f"Episode {episode} maps to multiple tasks")
        task_id = next(iter(tasks))
        for arm in ARMS:
            summary = _summarize_arm(
                payload, gate, states[arm], arm, mask
            )
            episode_rows.append(
                _table_row(
                    identity,
                    arm,
                    summary,
                    {
                        "episode_key": episode,
                        "task_id": task_id,
                    },
                )
            )

    task_rows = []
    for task_id in sorted(set(np.asarray(payload["task_id"]).astype(int).tolist())):
        mask = np.asarray(payload["task_id"]).astype(int) == task_id
        for arm in ARMS:
            summary = _summarize_arm(
                payload, gate, states[arm], arm, mask
            )
            task_rows.append(
                _table_row(
                    identity,
                    arm,
                    summary,
                    {
                        "task_id": int(task_id),
                        "episode_count": int(
                            len(set(payload["episode_key"][mask].tolist()))
                        ),
                    },
                )
            )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / f"{split_name}_per_episode.csv"
    task_path = output_dir / f"{split_name}_per_task.csv"
    metrics_path = output_dir / f"{split_name}.json"
    _write_csv(episode_path, episode_rows)
    _write_csv(task_path, task_rows)
    result = {
        "schema": 1,
        "protocol": PROTOCOL,
        "target_definition": TARGET_DEFINITION,
        "model": identity["model"],
        "seed": identity["seed"],
        "split": split_name,
        "primary_metric": "active_formula_mae",
        "time_shift_tokens": int(time_shift_tokens),
        "prediction_artifact": {
            "path": payload["path"],
            "sha256": sha256_file(payload["path"]),
        },
        "gt_gate": {
            "path": gate["_path"],
            "sha256": gate["_sha256"],
        },
        "arms": arm_summaries,
        "arm_flattened_metrics": arm_flat,
        "flattened_metrics": _macro_flat(
            arm_flat["left"], arm_flat["right"]
        ),
        "per_episode_csv": str(episode_path),
        "per_task_csv": str(task_path),
        "interpretation_guard": (
            "Current-state q-shot direction dot current signed endpoint "
            "velocity is evaluated against independently differenced raw "
            "shoulder-to-hand distance; no future prediction is used."
        ),
    }
    atomic_json_dump(json_safe(result), metrics_path)
    result["metrics_path"] = str(metrics_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split")
    parser.add_argument("--time-shift-tokens", type=int, default=1)
    args = parser.parse_args()
    result = evaluate_prediction_file(
        args.predictions,
        args.gt_gate,
        args.output_dir,
        split_name=args.split,
        time_shift_tokens=args.time_shift_tokens,
    )
    print(
        json.dumps(
            json_safe(
                {
                    "metrics": result["metrics_path"],
                    "primary": result["flattened_metrics"][
                        "active_formula_mae"
                    ],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
