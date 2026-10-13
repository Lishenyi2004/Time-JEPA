#!/usr/bin/env python3
"""GT-grounded COVLA kinematic evaluation.

The original consistency evaluator asks whether independently decoded variables
agree with one another.  This module adds the complementary, non-degenerate
question: do decoded trajectories and decoded-state rollouts agree with the
*measured COVLA trajectory*?

No encoder or probe is trained here.  The evaluator consumes an existing
``predictions/*.npz`` artifact and preserves the frozen SE(2) convention from
the train/val GT gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .bank_common import (
        PRIMARY_HORIZONS,
        TARGET_INDEX,
        TARGET_NAMES,
        atomic_json_dump,
        json_safe,
        sha256_file,
    )
    from .consistency_eval import (
        NEGATIVE_NAMES,
        _binary_discrimination,
        _capture_date,
        _motion_strata,
        _rollout,
        _sequence_components,
        _trajectory,
        _wrong_video_donors,
    )
    from .covla_gt_consistency_gate import integrate_scalar, mean_absolute_over_window
except ImportError:
    from bank_common import (
        PRIMARY_HORIZONS,
        TARGET_INDEX,
        TARGET_NAMES,
        atomic_json_dump,
        json_safe,
        sha256_file,
    )
    from consistency_eval import (
        NEGATIVE_NAMES,
        _binary_discrimination,
        _capture_date,
        _motion_strata,
        _rollout,
        _sequence_components,
        _trajectory,
        _wrong_video_donors,
    )
    from covla_gt_consistency_gate import integrate_scalar, mean_absolute_over_window


SCOPES = ("all", "gross_screened")
CONDITIONS = (
    "overall",
    "active_turning",
    "straight",
    "active_acceleration",
    "accelerating",
    "braking",
    "low_speed",
    "high_speed",
    "active_lateral",
)
GROSS_ENDPOINT_SPEED_M_PER_S = 40.0


def _vector_summary(delta: np.ndarray, mask: np.ndarray) -> dict:
    """Summarize a 2-D displacement error on a predeclared mask."""

    delta = np.asarray(delta, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    finite = mask & np.all(np.isfinite(delta), axis=1)
    values = delta[finite]
    if not len(values):
        return {"count": 0, "sum_2d_m": 0.0}
    norms = np.linalg.norm(values, axis=1)
    return {
        "count": int(len(values)),
        "sum_2d_m": float(np.sum(norms)),
        "x_mae_m": float(np.mean(np.abs(values[:, 0]))),
        "y_mae_m": float(np.mean(np.abs(values[:, 1]))),
        "mae_2d_m": float(np.mean(norms)),
        "rmse_2d_m": float(math.sqrt(np.mean(norms**2))),
        "p50_2d_m": float(np.quantile(norms, 0.50)),
        "p90_2d_m": float(np.quantile(norms, 0.90)),
        "p95_2d_m": float(np.quantile(norms, 0.95)),
        "p99_2d_m": float(np.quantile(norms, 0.99)),
    }


def _condition_masks(
    gt_components: dict[str, np.ndarray],
    gt_trajectory: np.ndarray,
    dt: float,
    horizon: float,
    count: int,
    thresholds: dict,
) -> dict[str, np.ndarray]:
    turn = mean_absolute_over_window(gt_components["yaw_rate"], dt, 1.0)[:count]
    acceleration = mean_absolute_over_window(gt_components["a_ego"], dt, 1.0)[:count]
    signed_acceleration = integrate_scalar(
        gt_components["a_ego"], dt, 1.0, "trapezoid"
    )[:count]
    speed = mean_absolute_over_window(gt_components["v_ego"], dt, 1.0)[:count]
    acceleration_threshold = thresholds["active_acceleration_q75_m_per_s2"]
    return {
        "overall": np.ones(count, dtype=bool),
        "active_turning": turn >= thresholds["active_turning_q75_rad_per_s"],
        "straight": turn < thresholds["active_turning_q75_rad_per_s"],
        "active_acceleration": acceleration >= acceleration_threshold,
        "accelerating": signed_acceleration >= acceleration_threshold,
        "braking": signed_acceleration <= -acceleration_threshold,
        "low_speed": speed <= thresholds["low_speed_q10_m_per_s"],
        "high_speed": speed >= thresholds["high_speed_q75_m_per_s"],
        "active_lateral": np.abs(gt_trajectory[:count, 1])
        >= thresholds["active_lateral_q75_m_by_horizon"][str(horizon)],
    }


def _video_grounded_record(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    target_frames: np.ndarray,
    video_fps: float,
    thresholds: dict,
    video_id: str,
    capture_date: str,
) -> dict:
    pred_components = _sequence_components(prediction, valid)
    gt_components = _sequence_components(target, valid)
    centers = np.mean(target_frames, axis=1)
    dt = float(np.median(np.diff(centers)) / video_fps)
    record = {
        "video_id": str(video_id),
        "capture_date": str(capture_date),
        "dt": dt,
        "horizons": {},
    }
    for horizon, point in PRIMARY_HORIZONS:
        key = str(horizon)
        pred_rollout = _rollout(pred_components, dt, horizon)
        gt_rollout = _rollout(gt_components, dt, horizon)
        pred_trajectory = _trajectory(prediction, valid, point)
        gt_trajectory = _trajectory(target, valid, point)
        count = min(
            len(pred_rollout),
            len(gt_rollout),
            len(pred_trajectory),
            len(gt_trajectory),
        )
        pred_rollout = pred_rollout[:count]
        gt_rollout = gt_rollout[:count]
        pred_trajectory = pred_trajectory[:count]
        gt_trajectory = gt_trajectory[:count]
        finite = (
            np.all(np.isfinite(pred_rollout), axis=1)
            & np.all(np.isfinite(gt_rollout), axis=1)
            & np.all(np.isfinite(pred_trajectory), axis=1)
            & np.all(np.isfinite(gt_trajectory), axis=1)
        )
        implied_speed = np.linalg.norm(gt_trajectory, axis=1) / float(horizon)
        gross_quality = implied_speed <= GROSS_ENDPOINT_SPEED_M_PER_S
        conditions = _condition_masks(
            gt_components, gt_trajectory, dt, horizon, count, thresholds
        )
        record["horizons"][key] = {
            "finite": finite,
            "gross_quality": gross_quality,
            "conditions": conditions,
            "direct_trajectory_to_gt": pred_trajectory - gt_trajectory,
            "rollout_to_gt_trajectory": pred_rollout - gt_trajectory,
            "rollout_to_gt_rollout": pred_rollout - gt_rollout,
            "gt_oracle_floor": gt_rollout - gt_trajectory,
            "gt_trajectory": gt_trajectory,
        }
    return record


def _scope_mask(item: dict, scope: str) -> np.ndarray:
    if scope == "all":
        return item["finite"]
    if scope == "gross_screened":
        return item["finite"] & item["gross_quality"]
    raise ValueError(f"Unknown grounded scope: {scope}")


def _aggregate_grounded(records: list[dict]) -> dict:
    output = {scope: {} for scope in SCOPES}
    metric_names = (
        "direct_trajectory_to_gt",
        "rollout_to_gt_trajectory",
        "rollout_to_gt_rollout",
        "gt_oracle_floor",
    )
    for scope in SCOPES:
        for horizon, _ in PRIMARY_HORIZONS:
            key = str(horizon)
            condition_output = {}
            for condition in CONDITIONS:
                masks, trajectory_chunks = [], []
                metric_chunks = {name: [] for name in metric_names}
                potential = 0
                gross_excluded = 0
                for record in records:
                    item = record["horizons"][key]
                    condition_mask = item["conditions"][condition]
                    base = item["finite"] & condition_mask
                    selected = _scope_mask(item, scope) & condition_mask
                    potential += int(np.sum(condition_mask))
                    gross_excluded += int(np.sum(base & ~item["gross_quality"]))
                    masks.append(selected)
                    trajectory_chunks.append(item["gt_trajectory"])
                    for name in metric_names:
                        metric_chunks[name].append(item[name])
                mask = np.concatenate(masks)
                trajectories = np.concatenate(trajectory_chunks)
                summary = {
                    "count": int(np.sum(mask)),
                    "potential": int(potential),
                    "valid_rate": float(np.sum(mask) / max(potential, 1)),
                    "gross_excluded_count": int(gross_excluded),
                    "gt_path_mean_norm_m": float(
                        np.mean(np.linalg.norm(trajectories[mask], axis=1))
                    )
                    if np.any(mask)
                    else None,
                }
                for name in metric_names:
                    summary[name] = _vector_summary(
                        np.concatenate(metric_chunks[name]), mask
                    )
                condition_output[condition] = summary
            output[scope][key] = {"conditions": condition_output}
    return output


def _per_video_rows(records: list[dict]) -> list[dict]:
    rows = []
    metric_names = (
        "direct_trajectory_to_gt",
        "rollout_to_gt_trajectory",
        "rollout_to_gt_rollout",
        "gt_oracle_floor",
    )
    for record in records:
        for horizon, _ in PRIMARY_HORIZONS:
            key = str(horizon)
            item = record["horizons"][key]
            for scope in SCOPES:
                mask = _scope_mask(item, scope)
                row = {
                    "video_id": record["video_id"],
                    "capture_date": record["capture_date"],
                    "horizon_seconds": float(horizon),
                    "scope": scope,
                    "count": int(np.sum(mask)),
                    "potential": int(len(mask)),
                    "gross_excluded_count": int(
                        np.sum(item["finite"] & ~item["gross_quality"])
                    ),
                }
                for name in metric_names:
                    values = np.linalg.norm(item[name][mask], axis=1)
                    row[f"{name}_sum_m"] = float(np.sum(values))
                    row[f"{name}_mae_2d_m"] = (
                        float(np.mean(values)) if len(values) else float("nan")
                    )
                target_norm = np.linalg.norm(item["gt_trajectory"][mask], axis=1)
                row["gt_path_norm_sum_m"] = float(np.sum(target_norm))
                row["gt_path_mean_norm_m"] = (
                    float(np.mean(target_norm)) if len(target_norm) else float("nan")
                )
                rows.append(row)
    return rows


def _summarize_hard_negative(score_chunks: dict, condition_chunks: dict) -> dict:
    scores = {
        name: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        for name, chunks in score_chunks.items()
    }
    correct = scores["correct"]
    if not len(correct):
        return {"count": 0}
    negatives = np.stack([scores[name] for name in NEGATIVE_NAMES], axis=1)
    conditions = {
        name: np.concatenate(chunks) if chunks else np.zeros(len(correct), dtype=bool)
        for name, chunks in condition_chunks.items()
    }
    rank = 1.0 + np.sum(negatives < correct[:, None], axis=1)
    rank += 0.5 * np.sum(negatives == correct[:, None], axis=1)
    all_candidates = np.concatenate([correct[:, None], negatives], axis=1)
    minima = np.min(all_candidates, axis=1, keepdims=True)
    tied_minimum = np.isclose(all_candidates, minima, rtol=1e-12, atol=1e-12)
    tie_aware_top1 = tied_minimum[:, 0] / np.sum(tied_minimum, axis=1)
    recommended_condition = {
        "wrong_yaw_sign": "active_turning",
        "wrong_vy_sign": "active_lateral",
        "one_token_time_shift": "active_acceleration",
        "matched_wrong_video": "overall",
        "wrong_horizon": "overall",
    }
    pairwise = {}
    conditioned_aurocs = []
    for column, name in enumerate(NEGATIVE_NAMES):
        condition_name = recommended_condition[name]
        selected = (
            np.ones(len(correct), dtype=bool)
            if condition_name == "overall"
            else conditions[condition_name]
        )
        pairwise[name] = {
            "recommended_condition": condition_name,
            "overall": _binary_discrimination(correct, negatives[:, column]),
            "conditioned": _binary_discrimination(
                correct[selected], negatives[selected, column]
            ),
        }
        conditioned = pairwise[name]["conditioned"]
        if conditioned.get("count", 0) > 0:
            conditioned_aurocs.append(conditioned["auroc"])
    return {
        "count": int(len(correct)),
        "strict_recall_at_1": float(
            np.mean((correct[:, None] < negatives).all(axis=1))
        ),
        "tie_aware_expected_recall_at_1": float(np.mean(tie_aware_top1)),
        "midrank_mrr": float(np.mean(1.0 / rank)),
        "conditioned_macro_auroc": float(np.mean(conditioned_aurocs)),
        "mean_margin_hardest_negative_minus_correct_m": float(
            np.mean(np.min(negatives, axis=1) - correct)
        ),
        "pairwise": pairwise,
    }


def _grounded_hard_negative_scores(
    prediction: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    target_valid: np.ndarray,
    frames: np.ndarray,
    video_fps: float,
    thresholds: dict,
    donors: np.ndarray,
) -> dict:
    score_chunks = {
        scope: {
            str(horizon): {name: [] for name in ("correct",) + NEGATIVE_NAMES}
            for horizon, _ in PRIMARY_HORIZONS
        }
        for scope in SCOPES
    }
    condition_chunks = {
        scope: {
            str(horizon): {
                "active_turning": [],
                "active_lateral": [],
                "active_acceleration": [],
            }
            for horizon, _ in PRIMARY_HORIZONS
        }
        for scope in SCOPES
    }
    for video_index in range(len(prediction)):
        components = _sequence_components(prediction[video_index], valid[video_index])
        donor_components = _sequence_components(
            prediction[donors[video_index]], valid[donors[video_index]]
        )
        gt_components = _sequence_components(
            target[video_index], target_valid[video_index]
        )
        dt = float(
            np.median(np.diff(np.mean(frames[video_index], axis=1))) / video_fps
        )
        for horizon_index, (horizon, point) in enumerate(PRIMARY_HORIZONS):
            key = str(horizon)
            gt_trajectory = _trajectory(
                target[video_index], target_valid[video_index], point
            )
            candidates = {
                "correct": _rollout(components, dt, horizon),
                "wrong_yaw_sign": _rollout(
                    components, dt, horizon, yaw_sign=1, vy_sign=-1
                ),
                "wrong_vy_sign": _rollout(
                    components, dt, horizon, yaw_sign=-1, vy_sign=1
                ),
                "one_token_time_shift": _rollout(
                    {name: value[1:] for name, value in components.items()},
                    dt,
                    horizon,
                ),
                "matched_wrong_video": _rollout(donor_components, dt, horizon),
            }
            alternate_horizon = PRIMARY_HORIZONS[
                (horizon_index + 1) % len(PRIMARY_HORIZONS)
            ][0]
            candidates["wrong_horizon"] = _rollout(
                components, dt, alternate_horizon
            )
            common = min([len(gt_trajectory)] + [len(item) for item in candidates.values()])
            if common <= 0:
                continue
            gt_trajectory = gt_trajectory[:common]
            distances = {
                name: np.linalg.norm(gt_trajectory - rollout[:common], axis=1)
                for name, rollout in candidates.items()
            }
            finite = np.all(np.isfinite(gt_trajectory), axis=1)
            for values in distances.values():
                finite &= np.isfinite(values)
            quality = (
                np.linalg.norm(gt_trajectory, axis=1) / float(horizon)
                <= GROSS_ENDPOINT_SPEED_M_PER_S
            )
            masks = _condition_masks(
                gt_components, gt_trajectory, dt, horizon, common, thresholds
            )
            selected_conditions = {
                "active_turning": masks["active_turning"],
                "active_lateral": masks["active_lateral"],
                "active_acceleration": masks["active_acceleration"],
            }
            for scope in SCOPES:
                selected = finite if scope == "all" else finite & quality
                for name, values in distances.items():
                    score_chunks[scope][key][name].append(values[selected])
                for name, condition in selected_conditions.items():
                    condition_chunks[scope][key][name].append(condition[selected])
    return {
        scope: {
            key: _summarize_hard_negative(
                score_chunks[scope][key], condition_chunks[scope][key]
            )
            for key in score_chunks[scope]
        }
        for scope in SCOPES
    }


def evaluate_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    target_frames: np.ndarray,
    video_ids: np.ndarray,
    video_fps: float,
    thresholds: dict,
    seed: int,
    capture_dates: dict[str, str] | None = None,
) -> tuple[dict, list[dict]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    target_frames = np.asarray(target_frames, dtype=np.float64)
    video_ids = np.asarray(video_ids).astype(str)
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target, and valid must have identical shapes")
    if prediction.ndim != 3 or prediction.shape[-1] != len(TARGET_NAMES):
        raise ValueError(f"Expected [N,T,{len(TARGET_NAMES)}], got {prediction.shape}")
    if len(video_ids) != len(prediction) or len(set(video_ids.tolist())) != len(video_ids):
        raise ValueError("video_ids must be unique and match prediction axis 0")
    capture_dates = capture_dates or {str(video_id): "unknown" for video_id in video_ids}
    records = [
        _video_grounded_record(
            prediction[index],
            target[index],
            valid[index],
            target_frames[index],
            video_fps,
            thresholds,
            str(video_id),
            capture_dates.get(str(video_id), "unknown"),
        )
        for index, video_id in enumerate(video_ids)
    ]
    labels, _ = _motion_strata(
        target, valid, target_frames, video_fps, thresholds
    )
    donors = _wrong_video_donors(labels, int(seed))
    grounded = _aggregate_grounded(records)
    hard_negative = _grounded_hard_negative_scores(
        prediction,
        valid,
        target,
        valid,
        target_frames,
        video_fps,
        thresholds,
        donors,
    )
    quality_audit = {}
    for horizon, _ in PRIMARY_HORIZONS:
        key = str(horizon)
        affected = []
        excluded = 0
        potential = 0
        for record in records:
            item = record["horizons"][key]
            bad = item["finite"] & ~item["gross_quality"]
            potential += int(np.sum(item["finite"]))
            excluded += int(np.sum(bad))
            if np.any(bad):
                affected.append(record["video_id"])
        quality_audit[key] = {
            "potential_samples": int(potential),
            "excluded_samples": int(excluded),
            "affected_video_count": int(len(affected)),
            "affected_video_ids": sorted(affected),
        }
    result = {
        "grounded_metrics": grounded,
        "gt_anchored_hard_negative": hard_negative,
        "gross_quality_audit": quality_audit,
    }
    return result, _per_video_rows(records)


def _validate_gate(gate: dict) -> dict:
    if gate.get("gate_decision", {}).get("status") != "PASS":
        raise RuntimeError("Frozen GT Gate is not PASS")
    selected = gate["coordinate_and_integrator_selection"]["selected"]
    expected = {
        "method": "trapezoid",
        "x_field": "velocities_calib[0]",
        "vy_sign": -1,
        "yaw_sign": -1,
        "trajectory_y_sign": 1,
    }
    for key, value in expected.items():
        if selected.get(key) != value:
            raise RuntimeError(
                f"GT Gate protocol drift for {key}: {selected.get(key)!r} != {value!r}"
            )
    return expected


def _load_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    target_names = tuple(payload["target_names"].astype(str).tolist())
    if target_names != TARGET_NAMES:
        raise ValueError(f"Target order mismatch: {target_names} != {TARGET_NAMES}")
    return payload


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_prediction_file(
    prediction_path: str | Path,
    gate_metrics_path: str | Path,
    output_dir: str | Path,
    seed: int | None = None,
) -> dict:
    prediction_path = Path(prediction_path).expanduser().resolve()
    gate_metrics_path = Path(gate_metrics_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    payload = _load_prediction(prediction_path)
    gate = json.loads(gate_metrics_path.read_text())
    frozen_protocol = _validate_gate(gate)
    video_ids = payload["video_id"].astype(str)
    capture_dates = {
        str(video_id): _capture_date(gate["metadata"]["covla_root"], str(video_id))
        for video_id in video_ids
    }
    stored_seed = int(payload["seed"].item())
    evaluated, rows = evaluate_arrays(
        payload["prediction"],
        payload["target"],
        payload["valid"],
        payload["target_frame_indices"],
        video_ids,
        float(payload["video_fps"].item()),
        gate["train_frozen_thresholds"],
        stored_seed if seed is None else int(seed),
        capture_dates,
    )
    split = str(payload["split_name"].item())
    model = str(payload["model_name"].item())
    result = {
        "schema": 1,
        "experiment": "covla_gt_grounded_kinematic_evaluation_v1",
        "model_name": model,
        "probe_seed": stored_seed,
        "split": split,
        "video_count": int(len(video_ids)),
        "definition": {
            "primary": "decoded-state SE(2) rollout is compared numerically with the measured COVLA trajectory endpoint",
            "direct": "independently decoded trajectory endpoint is compared numerically with the measured COVLA endpoint",
            "oracle": "GT-state SE(2) rollout versus the same measured trajectory endpoint",
            "gross_screen": (
                "data-only sensitivity excludes endpoint samples whose GT displacement / "
                f"horizon exceeds {GROSS_ENDPOINT_SPEED_M_PER_S:g} m/s; full frozen split remains primary"
            ),
        },
        "frozen_protocol": frozen_protocol,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "gt_gate_path": str(gate_metrics_path),
        "gt_gate_sha256": sha256_file(gate_metrics_path),
        **evaluated,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{split}_grounded.json"
    csv_path = output_dir / f"{split}_grounded_per_video.csv"
    result["outputs"] = {
        "json": str(json_path),
        "per_video_csv": str(csv_path),
    }
    atomic_json_dump(json_safe(result), json_path)
    _write_csv(csv_path, rows)
    return result


def evaluate_controls(
    reference_prediction_path: str | Path,
    gate_metrics_path: str | Path,
    output_path: str | Path,
    seed: int = 73000,
) -> dict:
    reference_prediction_path = Path(reference_prediction_path).expanduser().resolve()
    gate_metrics_path = Path(gate_metrics_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    payload = _load_prediction(reference_prediction_path)
    gate = json.loads(gate_metrics_path.read_text())
    _validate_gate(gate)
    video_ids = payload["video_id"].astype(str)
    capture_dates = {
        str(video_id): _capture_date(gate["metadata"]["covla_root"], str(video_id))
        for video_id in video_ids
    }
    target = payload["target"].astype(np.float64)
    controls = {
        "all_zero": np.zeros_like(target),
        "train_mean": np.broadcast_to(
            payload["target_mean"][None, None, :], target.shape
        ).copy(),
        "gt_oracle": target.copy(),
    }
    evaluated_controls = {}
    for offset, (name, prediction) in enumerate(controls.items()):
        evaluated, _ = evaluate_arrays(
            prediction,
            target,
            payload["valid"],
            payload["target_frame_indices"],
            video_ids,
            float(payload["video_fps"].item()),
            gate["train_frozen_thresholds"],
            int(seed) + offset,
            capture_dates,
        )
        evaluated_controls[name] = evaluated
    result = {
        "schema": 1,
        "experiment": "covla_gt_grounded_degenerate_controls_v1",
        "split": str(payload["split_name"].item()),
        "reference_prediction_path": str(reference_prediction_path),
        "reference_prediction_sha256": sha256_file(reference_prediction_path),
        "controls": evaluated_controls,
    }
    atomic_json_dump(json_safe(result), output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    result = evaluate_prediction_file(
        args.predictions, args.gt_gate, args.output_dir, args.seed
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "model": result["model_name"],
                "seed": result["probe_seed"],
                "split": result["split"],
                "output": result["outputs"]["json"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
