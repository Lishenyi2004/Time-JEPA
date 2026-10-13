#!/usr/bin/env python3
"""Train/val-only GT Gate for the AgiBot arm reach-rate identity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from physics_consistency.agibot.urdf_fk import (
    ARM_CHAINS,
    G2ForwardKinematics,
    apply_rigid_alignment,
    fit_rigid_alignment,
    position_error_summary,
    save_calibration,
)
from physics_consistency.agibot_reach_rate.common import (
    ARMS,
    P_SLICES,
    PROTOCOL,
    Q_INDICES,
    TARGET_DEFINITION,
    V_INDICES,
    atomic_json_dump,
    canonical_hash,
    json_safe,
    load_yaml,
    sha256_file,
)
from physics_consistency.agibot_reach_rate.gate_data import (
    collect_qp_targets,
    collect_reach_targets,
    load_gate_rows,
)
from physics_consistency.agibot_reach_rate.metrics import (
    direction_angle_metrics,
    regression_metrics,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _unit_direction(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector, axis=-1)
    valid = np.all(np.isfinite(vector), axis=-1) & (norm > 1e-8)
    output = np.zeros_like(vector)
    output[valid] = vector[valid] / norm[valid, None]
    return output, valid


def _aligned_fk(
    fk: G2ForwardKinematics,
    q: np.ndarray,
    arm: str,
    calibration: dict,
) -> np.ndarray:
    return apply_rigid_alignment(
        fk.forward(np.asarray(q, dtype=np.float64), arm),
        calibration,
    )


def _arm_arrays_qp(payload: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    q_slice = slice(0, 7) if arm == "left" else slice(7, 14)
    p_slice = slice(14, 17) if arm == "left" else slice(17, 20)
    q = np.asarray(payload["target"][:, q_slice], dtype=np.float64)
    p = np.asarray(payload["target"][:, p_slice], dtype=np.float64)
    valid = np.all(payload["valid"][:, q_slice] > 0, axis=-1) & np.all(
        payload["valid"][:, p_slice] > 0, axis=-1
    )
    return q[valid], p[valid]


def _shoulder_point(
    fk: G2ForwardKinematics,
    arm: str,
    calibration: dict,
) -> tuple[np.ndarray, np.ndarray]:
    urdf_point = np.asarray(
        fk.joints[ARM_CHAINS[arm][0]].origin[:3, 3],
        dtype=np.float64,
    )
    dataset_point = apply_rigid_alignment(
        urdf_point[None, :], calibration
    )[0]
    return urdf_point, dataset_point


def _train_reach_stats(payload: dict, arm_index: int) -> dict:
    values = np.asarray(payload["reach_rate_gt"][:, arm_index], dtype=np.float64)
    valid = (
        np.asarray(payload["reach_rate_valid"][:, arm_index]) > 0
    ) & np.isfinite(values)
    selected = values[valid]
    if len(selected) < 2:
        raise RuntimeError("Insufficient valid reach-rate values for frozen statistics")
    return {
        "count": int(len(selected)),
        "mean_mps": float(np.mean(selected)),
        "std_mps": float(np.std(selected, ddof=1)),
        "mean_abs_mps": float(np.mean(np.abs(selected))),
        "active_abs_q75_mps": float(np.quantile(np.abs(selected), 0.75)),
    }


def _formula_state(
    payload: dict,
    fk: G2ForwardKinematics,
    calibrations: dict,
    shoulder_points: dict,
    arm: str,
) -> dict:
    arm_index = ARMS.index(arm)
    other = "right" if arm == "left" else "left"
    q = np.asarray(payload["target"][:, Q_INDICES[arm]], dtype=np.float64)
    other_q = np.asarray(
        payload["target"][:, Q_INDICES[other]], dtype=np.float64
    )
    velocity = np.asarray(
        payload["target"][:, V_INDICES[arm]], dtype=np.float64
    )
    endpoint = np.asarray(
        payload["endpoint_position"][:, P_SLICES[arm]], dtype=np.float64
    )
    reach_rate = np.asarray(
        payload["reach_rate_gt"][:, arm_index], dtype=np.float64
    )
    q_valid = np.all(payload["valid"][:, Q_INDICES[arm]] > 0, axis=-1)
    other_q_valid = np.all(
        payload["valid"][:, Q_INDICES[other]] > 0, axis=-1
    )
    velocity_valid = np.all(
        payload["valid"][:, V_INDICES[arm]] > 0, axis=-1
    )
    endpoint_valid = payload["endpoint_valid"][:, arm_index] > 0
    reach_valid = payload["reach_rate_valid"][:, arm_index] > 0

    fk_endpoint = _aligned_fk(fk, q, arm, calibrations[arm])
    wrong_fk_endpoint = _aligned_fk(
        fk, other_q, arm, calibrations[arm]
    )
    shoulder = np.asarray(shoulder_points[arm], dtype=np.float64)
    raw_direction, raw_direction_valid = _unit_direction(
        endpoint - shoulder[None, :]
    )
    fk_direction, fk_direction_valid = _unit_direction(
        fk_endpoint - shoulder[None, :]
    )
    wrong_direction, wrong_direction_valid = _unit_direction(
        wrong_fk_endpoint - shoulder[None, :]
    )
    raw_formula = np.sum(raw_direction * velocity, axis=-1)
    urdf_formula = np.sum(fk_direction * velocity, axis=-1)
    wrong_formula = np.sum(wrong_direction * velocity, axis=-1)
    return {
        "q": q,
        "velocity": velocity,
        "endpoint": endpoint,
        "reach_rate": reach_rate,
        "q_valid": q_valid,
        "velocity_valid": velocity_valid,
        "endpoint_valid": endpoint_valid,
        "reach_valid": reach_valid,
        "fk_endpoint": fk_endpoint,
        "raw_direction": raw_direction,
        "fk_direction": fk_direction,
        "raw_formula": raw_formula,
        "urdf_formula": urdf_formula,
        "wrong_formula": wrong_formula,
        "raw_valid": (
            velocity_valid
            & endpoint_valid
            & reach_valid
            & raw_direction_valid
        ),
        "urdf_valid": (
            q_valid
            & velocity_valid
            & reach_valid
            & fk_direction_valid
        ),
        "wrong_valid": (
            other_q_valid
            & velocity_valid
            & reach_valid
            & wrong_direction_valid
        ),
    }


def _summarize_arm(
    payload: dict,
    fk: G2ForwardKinematics,
    calibrations: dict,
    shoulder_points: dict,
    arm: str,
    train_stats: dict,
    row_mask: np.ndarray | None = None,
) -> dict:
    state = _formula_state(
        payload, fk, calibrations, shoulder_points, arm
    )
    if row_mask is None:
        row_mask = np.ones(len(state["reach_rate"]), dtype=bool)
    row_mask = np.asarray(row_mask, dtype=bool)
    common = state["urdf_valid"] & row_mask
    raw_common = state["raw_valid"] & row_mask
    wrong_common = state["wrong_valid"] & row_mask
    q_endpoint_valid = (
        state["q_valid"] & state["endpoint_valid"] & row_mask
    )
    fk_position = position_error_summary(
        state["fk_endpoint"][q_endpoint_valid],
        state["endpoint"][q_endpoint_valid],
    )
    raw_formula = regression_metrics(
        state["raw_formula"],
        state["reach_rate"],
        raw_common,
        train_stats["mean_mps"],
        train_stats["std_mps"],
        train_stats["mean_abs_mps"],
    )
    urdf_formula = regression_metrics(
        state["urdf_formula"],
        state["reach_rate"],
        common,
        train_stats["mean_mps"],
        train_stats["std_mps"],
        train_stats["mean_abs_mps"],
    )
    wrong_formula = regression_metrics(
        state["wrong_formula"],
        state["reach_rate"],
        wrong_common,
        train_stats["mean_mps"],
        train_stats["std_mps"],
        train_stats["mean_abs_mps"],
    )
    return {
        "valid_token_count": int(np.sum(row_mask)),
        "formula_valid_fraction": float(
            np.sum(common) / max(np.sum(row_mask), 1)
        ),
        "fk_position_to_raw_endpoint": fk_position,
        "fk_direction_to_raw_endpoint": direction_angle_metrics(
            state["fk_direction"],
            state["raw_direction"],
            q_endpoint_valid,
        ),
        "raw_position_direction_dot_velocity_identity": raw_formula,
        "urdf_q_direction_dot_velocity_oracle": urdf_formula,
        "wrong_arm_q_direction_control": wrong_formula,
    }


def _flatten_gate_row(
    split: str,
    task_id: int,
    arm: str,
    summary: dict,
) -> dict:
    return {
        "split": split,
        "task_id": int(task_id),
        "arm": arm,
        "valid_token_count": summary["valid_token_count"],
        "formula_valid_fraction": summary["formula_valid_fraction"],
        "fk_mean_m": summary["fk_position_to_raw_endpoint"].get(
            "mean_euclidean_m"
        ),
        "fk_p95_m": summary["fk_position_to_raw_endpoint"].get(
            "p95_euclidean_m"
        ),
        "direction_mean_deg": summary[
            "fk_direction_to_raw_endpoint"
        ].get("mean_deg"),
        "raw_identity_pearson": summary[
            "raw_position_direction_dot_velocity_identity"
        ].get("pearson"),
        "raw_identity_nmae": summary[
            "raw_position_direction_dot_velocity_identity"
        ].get("nmae_train_mean_abs"),
        "urdf_oracle_pearson": summary[
            "urdf_q_direction_dot_velocity_oracle"
        ].get("pearson"),
        "urdf_oracle_nmae": summary[
            "urdf_q_direction_dot_velocity_oracle"
        ].get("nmae_train_mean_abs"),
        "wrong_arm_nmae": summary["wrong_arm_q_direction_control"].get(
            "nmae_train_mean_abs"
        ),
    }


def run_gate(config: dict) -> dict:
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"GT Gate protocol must be {PROTOCOL!r}")
    rows = load_gate_rows(config)
    sampling = config["sampling"]
    progress_every = int(config.get("gate", {}).get("progress_every", 0))
    fk = G2ForwardKinematics(config["kinematics"]["urdf"])
    actual_urdf_sha = sha256_file(config["kinematics"]["urdf"])
    expected_urdf_sha = config["kinematics"].get("sha256")
    if expected_urdf_sha and actual_urdf_sha != expected_urdf_sha:
        raise RuntimeError(
            f"URDF SHA256 mismatch: {actual_urdf_sha} != {expected_urdf_sha}"
        )

    calibration_targets = collect_qp_targets(
        rows["selection"], sampling, progress_every
    )
    calibrations = {}
    shoulder_urdf, shoulder_dataset = {}, {}
    for arm in ARMS:
        q, p = _arm_arrays_qp(calibration_targets, arm)
        calibration = fit_rigid_alignment(fk.forward(q, arm), p)
        calibrations[arm] = calibration
        shoulder_urdf[arm], shoulder_dataset[arm] = _shoulder_point(
            fk, arm, calibration
        )

    threshold_payload = collect_reach_targets(
        rows["threshold"],
        sampling,
        shoulder_dataset,
        progress_every,
        "probe_train reach targets",
    )
    validation_payload = collect_reach_targets(
        rows["validation"],
        sampling,
        shoulder_dataset,
        progress_every,
        "probe_val reach targets",
    )

    train_stats = {
        arm: _train_reach_stats(threshold_payload, arm_index)
        for arm_index, arm in enumerate(ARMS)
    }
    threshold_summary = {
        arm: _summarize_arm(
            threshold_payload,
            fk,
            calibrations,
            shoulder_dataset,
            arm,
            train_stats[arm],
        )
        for arm in ARMS
    }
    validation_summary = {
        arm: _summarize_arm(
            validation_payload,
            fk,
            calibrations,
            shoulder_dataset,
            arm,
            train_stats[arm],
        )
        for arm in ARMS
    }

    thresholds = config["gate"]["thresholds"]
    checks = {}
    for arm in ARMS:
        summary = validation_summary[arm]
        fk_position = summary["fk_position_to_raw_endpoint"]
        raw_identity = summary[
            "raw_position_direction_dot_velocity_identity"
        ]
        urdf_oracle = summary["urdf_q_direction_dot_velocity_oracle"]
        wrong = summary["wrong_arm_q_direction_control"]
        arm_thresholds = thresholds[arm]
        checks[f"{arm}_valid_fraction"] = (
            summary["formula_valid_fraction"]
            >= float(arm_thresholds["min_valid_fraction"])
        )
        checks[f"{arm}_fk_mean"] = (
            fk_position.get("mean_euclidean_m", np.inf)
            <= float(arm_thresholds["max_fk_mean_m"])
        )
        checks[f"{arm}_fk_p95"] = (
            fk_position.get("p95_euclidean_m", np.inf)
            <= float(arm_thresholds["max_fk_p95_m"])
        )
        checks[f"{arm}_raw_identity_pearson"] = (
            raw_identity.get("pearson", -np.inf)
            >= float(arm_thresholds["min_raw_identity_pearson"])
        )
        checks[f"{arm}_raw_identity_nmae"] = (
            raw_identity.get("nmae_train_mean_abs", np.inf)
            <= float(arm_thresholds["max_raw_identity_nmae"])
        )
        checks[f"{arm}_urdf_oracle_pearson"] = (
            urdf_oracle.get("pearson", -np.inf)
            >= float(arm_thresholds["min_urdf_oracle_pearson"])
        )
        checks[f"{arm}_urdf_oracle_nmae"] = (
            urdf_oracle.get("nmae_train_mean_abs", np.inf)
            <= float(arm_thresholds["max_urdf_oracle_nmae"])
        )
        checks[f"{arm}_correct_beats_wrong_arm"] = (
            urdf_oracle.get("mae", np.inf) < wrong.get("mae", -np.inf)
        )

    output_dir = Path(config["output"]["dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "frame_calibration.json"
    save_calibration(calibrations, calibration_path)

    per_task_rows = []
    for split_name, payload in (
        ("probe_train", threshold_payload),
        ("probe_val", validation_payload),
    ):
        for task_id in sorted(set(payload["task_id"].tolist())):
            task_mask = payload["task_id"] == task_id
            for arm in ARMS:
                summary = _summarize_arm(
                    payload,
                    fk,
                    calibrations,
                    shoulder_dataset,
                    arm,
                    train_stats[arm],
                    task_mask,
                )
                per_task_rows.append(
                    _flatten_gate_row(split_name, task_id, arm, summary)
                )
    per_task_path = output_dir / "gt_gate_per_task.csv"
    _write_csv(per_task_path, per_task_rows)

    gate_fingerprint = {
        "protocol": PROTOCOL,
        "target_definition": TARGET_DEFINITION,
        "sampling": sampling,
        "selection_split": config.get("selection_split"),
        "threshold_split": config.get("threshold_split"),
        "validation_split": config.get("validation_split"),
        "selection_window_ids": sorted(
            row["video_id"] for row in rows["selection"]
        ),
        "threshold_window_ids": sorted(
            row["video_id"] for row in rows["threshold"]
        ),
        "validation_window_ids": sorted(
            row["video_id"] for row in rows["validation"]
        ),
        "urdf_sha256": actual_urdf_sha,
    }
    result = {
        "schema": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": PROTOCOL,
        "target_definition": TARGET_DEFINITION,
        "fingerprint": {
            "sha256": canonical_hash(gate_fingerprint),
            "payload": gate_fingerprint,
        },
        "splits": {
            "selection": {
                "name": config.get("selection_split", "encoder_train"),
                "window_count": len(rows["selection"]),
            },
            "threshold": {
                "name": config.get("threshold_split", "probe_train"),
                "window_count": len(rows["threshold"]),
            },
            "validation": {
                "name": config.get("validation_split", "probe_val"),
                "window_count": len(rows["validation"]),
            },
            "view": config["data"].get(
                "gate_view", "observation.images.top_head"
            ),
        },
        "urdf": {
            "path": str(Path(config["kinematics"]["urdf"]).resolve()),
            "sha256": actual_urdf_sha,
        },
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "shoulder_points_urdf": {
            arm: shoulder_urdf[arm].tolist() for arm in ARMS
        },
        "shoulder_points_dataset": {
            arm: shoulder_dataset[arm].tolist() for arm in ARMS
        },
        "arms": {
            arm: {
                "reach_rate_train_stats": train_stats[arm],
                "probe_train": threshold_summary[arm],
                "probe_val": validation_summary[arm],
                "calibration": {
                    "rotation": calibrations[arm]["rotation"].tolist(),
                    "translation": calibrations[arm]["translation"].tolist(),
                    "fit_count": calibrations[arm]["fit_count"],
                },
            }
            for arm in ARMS
        },
        "checks": checks,
        "per_task_csv": str(per_task_path),
        "interpretation_guard": (
            "This Gate uses encoder_train/probe_train/probe_val only; "
            "it validates current-state kinematics, not future prediction."
        ),
    }
    output_path = output_dir / "gt_gate_metrics.json"
    atomic_json_dump(json_safe(result), output_path)
    result["metrics_path"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=Path(__file__).resolve().parent / "configs" / "gt_gate.yaml",
    )
    args = parser.parse_args()
    result = run_gate(load_yaml(args.config))
    print(
        json.dumps(
            json_safe(
                {
                    "status": result["status"],
                    "checks": result["checks"],
                    "metrics": result["metrics_path"],
                }
            ),
            indent=2,
        )
    )
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
