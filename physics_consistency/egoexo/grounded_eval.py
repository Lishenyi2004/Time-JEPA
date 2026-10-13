#!/usr/bin/env python3
"""GT-grounded cosine-law evaluation for EgoExo arm predictions."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

try:
    from .bank_common import (
        ANGLE_INDEX,
        DISTANCE_INDEX,
        ID_VIEWS,
        OOD_VIEWS,
        TARGET_INDEX,
        TARGET_NAMES,
        TARGET_SPECS,
        atomic_json_dump,
        json_safe,
        sha256_file,
    )
    from .metrics import (
        error_summary,
        hard_negative_summary,
        scalar_regression_metrics,
    )
    from .pose_geometry import SIDES, cosine_law_distance, morphology_for_pose
except ImportError:
    from bank_common import (
        ANGLE_INDEX,
        DISTANCE_INDEX,
        ID_VIEWS,
        OOD_VIEWS,
        TARGET_INDEX,
        TARGET_NAMES,
        TARGET_SPECS,
        atomic_json_dump,
        json_safe,
        sha256_file,
    )
    from metrics import error_summary, hard_negative_summary, scalar_regression_metrics
    from pose_geometry import SIDES, cosine_law_distance, morphology_for_pose


HARD_NEGATIVES = ("supplementary_angle", "wrong_side_angle", "one_token_time_shift")
GT_GATE_PROTOCOL = "egoexo_take_calibrated_arm_cosine_law_v1"
GROUNDED_PROTOCOL = "egoexo_gt_grounded_arm_cosine_law_v1"


def load_prediction_file(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    names = tuple(str(item) for item in payload["target_names"].tolist())
    if names != TARGET_NAMES:
        raise ValueError(f"Prediction target schema mismatch: {names} != {TARGET_NAMES}")
    expected_shape = payload["prediction"].shape
    for key in ("prediction", "target", "valid"):
        if payload[key].shape != expected_shape or payload[key].ndim != 3:
            raise ValueError(f"{key} shape mismatch: {payload[key].shape} vs {expected_shape}")
    if expected_shape[-1] != len(TARGET_NAMES):
        raise ValueError(f"Expected {len(TARGET_NAMES)} targets, got {expected_shape}")
    rows = expected_shape[0]
    for key in (
        "take_uid",
        "window_key",
        "task_id",
        "participant_uid",
        "capture_uid",
        "view_key",
        "pose_path",
    ):
        if key not in payload or len(payload[key]) != rows:
            raise ValueError(f"Prediction artifact needs row metadata {key!r}")
    payload["path"] = str(path)
    return payload


def _morphology_arrays(payload: dict, pose_config: dict) -> dict:
    paths = payload["pose_path"].astype(str)
    unique = {}
    for path in sorted(set(paths.tolist())):
        unique[path] = morphology_for_pose(
            path,
            float(pose_config.get("fps", 30.0)),
            int(pose_config.get("min_views", 2)),
            int(pose_config.get("smooth_window", 5)),
        )
    return {
        side: {
            "proximal": np.asarray(
                [unique[path][side]["proximal_length_m"] for path in paths],
                dtype=np.float64,
            )[:, None],
            "distal": np.asarray(
                [unique[path][side]["distal_length_m"] for path in paths],
                dtype=np.float64,
            )[:, None],
        }
        for side in SIDES
    }


def _active_mask(theta: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    active = np.zeros(valid.shape, dtype=bool)
    pair_valid = valid[:, 1:] & valid[:, :-1]
    delta = np.abs(theta[:, 1:] - theta[:, :-1])
    active[:, 1:] = pair_valid & (delta >= float(threshold))
    return active


def _shift_tokens(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = np.full_like(values, np.nan, dtype=np.float64)
    shifted_valid = np.zeros_like(valid, dtype=bool)
    shifted[:, :-1] = values[:, 1:]
    shifted_valid[:, :-1] = valid[:, 1:]
    return shifted, shifted_valid


def _distance_error(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    shape = np.asarray(valid).shape
    prediction = np.broadcast_to(np.asarray(prediction, dtype=np.float64), shape)
    target = np.broadcast_to(np.asarray(target, dtype=np.float64), shape)
    output = np.full(shape, np.nan, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(prediction) & np.isfinite(target)
    output[mask] = np.abs(prediction[mask] - target[mask])
    return output


def _arm_state(payload: dict, gate: dict, morphology: dict, side: str) -> dict:
    angle_index, distance_index = ANGLE_INDEX[side], DISTANCE_INDEX[side]
    other = "right" if side == "left" else "left"
    other_angle_index = ANGLE_INDEX[other]
    theta_pred = payload["prediction"][..., angle_index].astype(np.float64)
    theta_gt = payload["target"][..., angle_index].astype(np.float64)
    distance_pred = payload["prediction"][..., distance_index].astype(np.float64)
    distance_gt = payload["target"][..., distance_index].astype(np.float64)
    theta_valid = payload["valid"][..., angle_index] > 0
    distance_valid = payload["valid"][..., distance_index] > 0
    valid = theta_valid & distance_valid
    proximal = morphology[side]["proximal"]
    distal = morphology[side]["distal"]

    physics_pred = cosine_law_distance(theta_pred, proximal, distal, clip_angle=True)
    physics_gt = cosine_law_distance(theta_gt, proximal, distal, clip_angle=True)
    train_mean_theta = float(payload["target_mean"][angle_index])
    mean_physics = cosine_law_distance(train_mean_theta, proximal, distal)
    zero_physics = cosine_law_distance(0.0, proximal, distal)

    global_lengths = gate["global_probe_train_morphology"][side]
    global_physics = cosine_law_distance(
        theta_pred,
        float(global_lengths["proximal_length_m"]),
        float(global_lengths["distal_length_m"]),
        clip_angle=True,
    )

    theta_clipped = np.clip(theta_pred, 0.0, np.pi)
    supplementary = cosine_law_distance(np.pi - theta_clipped, proximal, distal)
    other_theta = payload["prediction"][..., other_angle_index].astype(np.float64)
    other_valid = payload["valid"][..., other_angle_index] > 0
    wrong_side = cosine_law_distance(other_theta, proximal, distal, clip_angle=True)
    shifted_theta, shifted_valid = _shift_tokens(theta_pred, theta_valid)
    shifted = cosine_law_distance(shifted_theta, proximal, distal, clip_angle=True)
    scores = {
        "correct": _distance_error(physics_pred, distance_gt, valid),
        "supplementary_angle": _distance_error(supplementary, distance_gt, valid),
        "wrong_side_angle": _distance_error(
            wrong_side, distance_gt, valid & other_valid
        ),
        "one_token_time_shift": _distance_error(
            shifted, distance_gt, valid & shifted_valid
        ),
    }

    gt_theta_clipped = np.clip(theta_gt, 0.0, np.pi)
    gt_other_theta = payload["target"][..., other_angle_index].astype(np.float64)
    gt_shifted_theta, gt_shifted_valid = _shift_tokens(theta_gt, theta_valid)
    gt_scores = {
        "correct": _distance_error(physics_gt, distance_gt, valid),
        "supplementary_angle": _distance_error(
            cosine_law_distance(np.pi - gt_theta_clipped, proximal, distal),
            distance_gt,
            valid,
        ),
        "wrong_side_angle": _distance_error(
            cosine_law_distance(gt_other_theta, proximal, distal),
            distance_gt,
            valid & (payload["valid"][..., other_angle_index] > 0),
        ),
        "one_token_time_shift": _distance_error(
            cosine_law_distance(gt_shifted_theta, proximal, distal),
            distance_gt,
            valid & gt_shifted_valid,
        ),
    }
    threshold = float(gate["active_thresholds"][side])
    active = _active_mask(theta_gt, theta_valid, threshold) & distance_valid
    return {
        "theta_pred": theta_pred,
        "theta_gt": theta_gt,
        "distance_pred": distance_pred,
        "distance_gt": distance_gt,
        "theta_valid": theta_valid,
        "distance_valid": distance_valid,
        "valid": valid,
        "active": active,
        "physics_pred": physics_pred,
        "physics_gt": physics_gt,
        "global_physics": global_physics,
        "mean_physics": mean_physics,
        "zero_physics": zero_physics,
        "scores": scores,
        "gt_scores": gt_scores,
        "active_threshold": threshold,
    }


def _masked_scores(scores: dict[str, np.ndarray], row_mask: np.ndarray) -> dict:
    return {name: values[row_mask] for name, values in scores.items()}


def _summarize_arm(
    payload: dict, state: dict, side: str, row_mask: np.ndarray
) -> dict:
    angle_index, distance_index = ANGLE_INDEX[side], DISTANCE_INDEX[side]
    valid = state["valid"][row_mask]
    active = state["active"][row_mask]
    physics = error_summary(
        state["physics_pred"][row_mask], state["distance_gt"][row_mask], valid
    )
    direct = error_summary(
        state["distance_pred"][row_mask],
        state["distance_gt"][row_mask],
        state["distance_valid"][row_mask],
    )
    oracle = error_summary(
        state["physics_gt"][row_mask], state["distance_gt"][row_mask], valid
    )
    train_mean = error_summary(
        state["mean_physics"][row_mask], state["distance_gt"][row_mask], valid
    )
    zero = error_summary(
        state["zero_physics"][row_mask], state["distance_gt"][row_mask], valid
    )
    global_morphology = error_summary(
        state["global_physics"][row_mask], state["distance_gt"][row_mask], valid
    )
    closure = error_summary(
        state["physics_pred"][row_mask], state["distance_pred"][row_mask], valid
    )
    active_physics = error_summary(
        state["physics_pred"][row_mask], state["distance_gt"][row_mask], active
    )
    model_sse = np.nansum(state["scores"]["correct"][row_mask] ** 2)
    baseline_error = _distance_error(
        state["mean_physics"][row_mask], state["distance_gt"][row_mask], valid
    )
    baseline_sse = np.nansum(baseline_error**2)
    raw_theta = state["theta_pred"][row_mask]
    theta_mask = state["theta_valid"][row_mask] & np.isfinite(raw_theta)
    angle_point = scalar_regression_metrics(
        raw_theta,
        state["theta_gt"][row_mask],
        state["theta_valid"][row_mask],
        float(payload["target_mean"][angle_index]),
        float(payload["target_std"][angle_index]),
        "rad",
    )
    angle_point["physical_range_violation_rate"] = (
        float(np.mean((raw_theta[theta_mask] < 0.0) | (raw_theta[theta_mask] > np.pi)))
        if np.any(theta_mask)
        else np.nan
    )
    return {
        "angle_point": angle_point,
        "direct_distance_point": scalar_regression_metrics(
            state["distance_pred"][row_mask],
            state["distance_gt"][row_mask],
            state["distance_valid"][row_mask],
            float(payload["target_mean"][distance_index]),
            float(payload["target_std"][distance_index]),
            "m",
        ),
        "grounded_physics_to_gt": physics,
        "grounded_physics_to_gt_active_flexion": active_physics,
        "direct_distance_to_gt": direct,
        "gt_angle_oracle_floor": oracle,
        "prediction_only_closure_diagnostic": closure,
        "controls": {
            "train_mean_angle_to_gt": train_mean,
            "all_zero_angle_to_gt": zero,
            "train_global_morphology_sensitivity": global_morphology,
            "physics_skill_vs_train_mean_angle": (
                float(1.0 - model_sse / baseline_sse) if baseline_sse > 0 else np.nan
            ),
        },
        "hard_negatives": hard_negative_summary(
            _masked_scores(state["scores"], row_mask), active
        ),
        "gt_angle_hard_negative_ceiling": hard_negative_summary(
            _masked_scores(state["gt_scores"], row_mask), active
        ),
        "active_threshold_token_delta_rad": state["active_threshold"],
    }


def _value(summary: dict, path: str, default=np.nan):
    current = summary
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _flatten_take_row(prefix: dict, summary: dict) -> dict:
    return {
        **prefix,
        "angle_nrmse": _value(summary, "angle_point.train_std_nrmse"),
        "angle_pearson": _value(summary, "angle_point.pearson"),
        "direct_distance_mae_m": _value(summary, "direct_distance_to_gt.mae_m"),
        "physics_mae_m": _value(summary, "grounded_physics_to_gt.mae_m"),
        "physics_active_mae_m": _value(
            summary, "grounded_physics_to_gt_active_flexion.mae_m"
        ),
        "physics_p95_m": _value(summary, "grounded_physics_to_gt.p95_m"),
        "oracle_mae_m": _value(summary, "gt_angle_oracle_floor.mae_m"),
        "closure_mae_m": _value(summary, "prediction_only_closure_diagnostic.mae_m"),
        "train_mean_control_mae_m": _value(
            summary, "controls.train_mean_angle_to_gt.mae_m"
        ),
        "zero_control_mae_m": _value(summary, "controls.all_zero_angle_to_gt.mae_m"),
        "global_morphology_mae_m": _value(
            summary, "controls.train_global_morphology_sensitivity.mae_m"
        ),
        "physics_skill_vs_train_mean": _value(
            summary, "controls.physics_skill_vs_train_mean_angle"
        ),
        "law_overall_macro_auroc": _value(
            summary, "hard_negatives.overall_macro_auroc"
        ),
        "law_conditioned_macro_auroc": _value(
            summary, "hard_negatives.conditioned_macro_auroc"
        ),
        "law_strict_recall_at_1": _value(summary, "hard_negatives.strict_recall_at_1"),
        "valid_tokens": _value(summary, "grounded_physics_to_gt.count", 0),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_prediction_file(
    prediction_path: str | Path,
    gt_gate_metrics: str | Path,
    output_dir: str | Path,
    pose_config: dict,
    split_name: str | None = None,
) -> dict:
    payload = load_prediction_file(prediction_path)
    gate_path = Path(gt_gate_metrics).expanduser().resolve()
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "PASS":
        raise RuntimeError(f"GT gate did not pass: {gate_path}")
    if gate.get("protocol") != GT_GATE_PROTOCOL:
        raise RuntimeError("Unexpected or obsolete EgoExo GT gate protocol")
    if split_name is None:
        split_name = str(payload.get("split_name", np.asarray("unknown")).item())
    morphology = _morphology_arrays(payload, pose_config)
    states = {side: _arm_state(payload, gate, morphology, side) for side in SIDES}
    row_count = payload["prediction"].shape[0]
    all_rows = np.ones(row_count, dtype=bool)
    overall = {
        side: _summarize_arm(payload, states[side], side, all_rows) for side in SIDES
    }
    per_take_rows = []
    take_values = payload["take_uid"].astype(str)
    for take_uid in sorted(set(take_values.tolist())):
        mask = take_values == take_uid
        first = int(np.flatnonzero(mask)[0])
        for side in SIDES:
            summary = _summarize_arm(payload, states[side], side, mask)
            per_take_rows.append(
                _flatten_take_row(
                    {
                        "split": split_name,
                        "view": str(payload["view_key"][first]),
                        "take_uid": take_uid,
                        "participant_uid": str(payload["participant_uid"][first]),
                        "capture_uid": str(payload["capture_uid"][first]),
                        "task_id": str(payload["task_id"][first]),
                        "side": side,
                        "window_count": int(np.sum(mask)),
                    },
                    summary,
                )
            )
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_take_path = output_dir / f"{split_name}_per_take.csv"
    _write_csv(per_take_path, per_take_rows)
    result = {
        "schema": 1,
        "protocol": GROUNDED_PROTOCOL,
        "split": split_name,
        "view_keys": sorted(set(payload["view_key"].astype(str).tolist())),
        "row_count": int(row_count),
        "token_count": int(np.prod(payload["prediction"].shape[:2])),
        "prediction_artifact": {
            "path": payload["path"],
            "sha256": sha256_file(payload["path"]),
        },
        "gt_gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
        "target_manifest": [dict(spec) for spec in TARGET_SPECS],
        "arms": overall,
        "per_take_csv": str(per_take_path),
        "interpretation_guard": (
            "grounded_physics_to_gt is primary; prediction_only_closure_diagnostic "
            "and view agreement are supplementary and cannot establish human dynamics."
        ),
    }
    metrics_path = output_dir / f"{split_name}_grounded.json"
    atomic_json_dump(json_safe(result), metrics_path)
    result["metrics_path"] = str(metrics_path)
    return result


def _align_payloads(payloads: dict[str, dict]) -> dict[str, dict]:
    aligned = {}
    reference_keys = None
    for view, payload in payloads.items():
        order = np.argsort(payload["window_key"].astype(str))
        current = {
            key: (
                value[order]
                if isinstance(value, np.ndarray)
                and value.ndim > 0
                and len(value) == len(order)
                else value
            )
            for key, value in payload.items()
        }
        keys = current["window_key"].astype(str)
        if reference_keys is None:
            reference_keys = keys
        elif not np.array_equal(keys, reference_keys):
            raise RuntimeError(f"Cross-view physical windows differ for {view}")
        aligned[view] = current
    return aligned


def evaluate_cross_view(
    prediction_paths: dict[str, str | Path],
    gt_gate_metrics: str | Path,
    pose_config: dict,
    output_path: str | Path,
) -> dict:
    payloads = {
        split_name.split("__", 1)[-1]: load_prediction_file(path)
        for split_name, path in prediction_paths.items()
    }
    payloads = _align_payloads(payloads)
    available_id = [view for view in ID_VIEWS if view in payloads]
    available_ood = [view for view in OOD_VIEWS if view in payloads]
    pairwise_rows = []
    for left_view, right_view in itertools.combinations(available_id + available_ood, 2):
        left, right = payloads[left_view], payloads[right_view]
        for index, name in enumerate(TARGET_NAMES):
            valid = (left["valid"][..., index] > 0) & (right["valid"][..., index] > 0)
            delta = np.abs(left["prediction"][..., index] - right["prediction"][..., index])
            pairwise_rows.append(
                {
                    "left_view": left_view,
                    "right_view": right_view,
                    "quantity": name,
                    "count": int(np.sum(valid)),
                    "pairwise_prediction_mae": float(np.mean(delta[valid])) if np.any(valid) else np.nan,
                }
            )

    consensus = None
    if available_id:
        base = payloads[available_id[0]]
        predictions = np.stack([payloads[view]["prediction"] for view in available_id], axis=0)
        valids = np.stack([payloads[view]["valid"] > 0 for view in available_id], axis=0)
        weighted = np.where(valids, predictions, 0.0).sum(axis=0)
        counts = valids.sum(axis=0)
        consensus_prediction = weighted / np.maximum(counts, 1)
        consensus_payload = dict(base)
        consensus_payload["prediction"] = consensus_prediction
        consensus_payload["valid"] = (counts > 0).astype(np.float32) * (base["valid"] > 0)
        gate = json.loads(Path(gt_gate_metrics).expanduser().resolve().read_text())
        morphology = _morphology_arrays(consensus_payload, pose_config)
        consensus = {
            side: _summarize_arm(
                consensus_payload,
                _arm_state(consensus_payload, gate, morphology, side),
                side,
                np.ones(len(base["window_key"]), dtype=bool),
            )
            for side in SIDES
        }
        if available_ood:
            ood = payloads[available_ood[0]]
            consensus["ood_vs_id_consensus"] = {}
            for index, name in enumerate(TARGET_NAMES):
                valid = (consensus_payload["valid"][..., index] > 0) & (
                    ood["valid"][..., index] > 0
                )
                delta = np.abs(
                    consensus_prediction[..., index] - ood["prediction"][..., index]
                )
                consensus["ood_vs_id_consensus"][name] = {
                    "count": int(np.sum(valid)),
                    "mae": float(np.mean(delta[valid])) if np.any(valid) else np.nan,
                }
    result = {
        "schema": 1,
        "protocol": "egoexo_paired_view_diagnostic_v1",
        "id_views": available_id,
        "ood_views": available_ood,
        "pairwise": pairwise_rows,
        "id_consensus_grounded": consensus,
        "interpretation_guard": (
            "Cross-view agreement is never used without grounded point/physics error; "
            "constant predictions can otherwise look perfectly invariant."
        ),
    }
    output_path = Path(output_path).expanduser().resolve()
    atomic_json_dump(json_safe(result), output_path)
    result["path"] = str(output_path)
    return result


def summarize_run(
    split_results: dict[str, dict],
    cross_view: dict | None,
    output_path: str | Path,
) -> dict:
    role_summary = {}
    for role in ("id", "ood"):
        selected = [
            result for name, result in split_results.items() if name.startswith(f"{role}__")
        ]
        if not selected:
            continue
        role_summary[role] = {}
        for side in SIDES:
            role_summary[role][side] = {}
            paths = {
                "physics_mae_m": "grounded_physics_to_gt.mae_m",
                "physics_active_mae_m": "grounded_physics_to_gt_active_flexion.mae_m",
                "direct_distance_mae_m": "direct_distance_to_gt.mae_m",
                "oracle_mae_m": "gt_angle_oracle_floor.mae_m",
                "law_overall_macro_auroc": "hard_negatives.overall_macro_auroc",
                "law_conditioned_macro_auroc": "hard_negatives.conditioned_macro_auroc",
                "law_strict_recall_at_1": "hard_negatives.strict_recall_at_1",
            }
            for name, path in paths.items():
                values = [_value(result["arms"][side], path) for result in selected]
                role_summary[role][side][name] = float(np.nanmean(values))
            role_summary[role][side]["view_count"] = len(selected)
    result = {
        "schema": 1,
        "protocol": "egoexo_run_grounded_summary_v1",
        "roles": role_summary,
        "split_metrics": {
            name: result["metrics_path"] for name, result in split_results.items()
        },
        "cross_view": cross_view.get("path") if cross_view else None,
    }
    output_path = Path(output_path).expanduser().resolve()
    atomic_json_dump(json_safe(result), output_path)
    result["path"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pose-fps", type=float, default=30.0)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--split-name")
    args = parser.parse_args()
    result = evaluate_prediction_file(
        args.predictions,
        args.gt_gate,
        args.output_dir,
        {
            "fps": args.pose_fps,
            "min_views": args.min_views,
            "smooth_window": args.smooth_window,
        },
        split_name=args.split_name,
    )
    print(json.dumps(json_safe({"metrics": result["metrics_path"], "arms": result["arms"]}), indent=2))


if __name__ == "__main__":
    main()
