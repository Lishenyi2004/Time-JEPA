#!/usr/bin/env python3
"""Freeze and validate the EgoExo articulated-arm cosine-law protocol on GT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

try:
    from .bank_common import atomic_json_dump, json_safe, sha256_file
    from .metrics import binary_discrimination, error_summary
    from .pose_geometry import (
        SIDES,
        cosine_law_distance,
        load_take_arm_geometry,
        pool_window_arm_targets,
    )
except ImportError:
    from bank_common import atomic_json_dump, json_safe, sha256_file
    from metrics import binary_discrimination, error_summary
    from pose_geometry import (
        SIDES,
        cosine_law_distance,
        load_take_arm_geometry,
        pool_window_arm_targets,
    )

from src.time_jepa.time_sampling import build_sample_times


GT_GATE_PROTOCOL = "egoexo_take_calibrated_arm_cosine_law_v1"
FORMULA_NAME = "law_of_cosines_internal_elbow_angle"
FORMULA_EXPRESSION = (
    "sqrt(L_upper^2 + L_fore^2 - 2*L_upper*L_fore*cos(theta))"
)
MORPHOLOGY_DESCRIPTION = (
    "per-take temporal median after 5-frame joint smoothing"
)
INTERPRETATION_GUARD = (
    "This gate validates pose-derived articulated geometry, not independent "
    "sensors, forces, contacts, or Newtonian human dynamics."
)


def _load_yaml(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).expanduser().resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"GT gate config must be a YAML mapping: {path}")
    return config, path


def _parse_window_key(key: str) -> tuple[str, int]:
    take_uid, suffix = str(key).rsplit("/window_", 1)
    return take_uid, int(suffix)


def _load_split(config: dict, split_name: str) -> tuple[list[str], Path]:
    if split_name in {"test", "weather_ood_test", "ood_test"}:
        raise ValueError("GT gate is forbidden from reading final test splits")
    data = config["data"]
    path = (
        Path(data["split_dir"]).expanduser().resolve()
        / f"{data['split_prefix']}_{split_name}.json"
    )
    with path.open() as handle:
        keys = json.load(handle)
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"Split is empty or malformed: {path}")
    limit = int(data.get("limits", {}).get(split_name, 0) or 0)
    if limit > 0:
        keys = keys[:limit]
    return [str(item) for item in keys], path


def _load_manifest(config: dict) -> tuple[dict, Path]:
    path = Path(config["data"]["manifest_path"]).expanduser().resolve()
    with path.open() as handle:
        manifest = json.load(handle)
    expected_root = Path(config["data"]["egoexo_root"]).expanduser().resolve()
    actual_root = Path(manifest["root"]).expanduser().resolve()
    if expected_root != actual_root:
        raise RuntimeError(f"Manifest root mismatch: {actual_root} != {expected_root}")
    return manifest, path


def _active_threshold(theta: np.ndarray, valid: np.ndarray, quantile: float) -> float:
    pair_valid = valid[:, 1:] & valid[:, :-1]
    delta = np.abs(theta[:, 1:] - theta[:, :-1])
    values = delta[pair_valid & np.isfinite(delta)]
    if not len(values):
        raise RuntimeError("No valid adjacent angle pairs for active threshold")
    return float(np.quantile(values, float(quantile)))


def _active_mask(theta: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    active = np.zeros(valid.shape, dtype=bool)
    pair_valid = valid[:, 1:] & valid[:, :-1]
    delta = np.abs(theta[:, 1:] - theta[:, :-1])
    active[:, 1:] = pair_valid & (delta >= float(threshold))
    return active


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _collect_split(
    config: dict,
    manifest: dict,
    split_name: str,
    keys: list[str],
) -> dict:
    sampling = config["sampling"]
    pose = config["pose"]
    tokens = int(sampling["num_frames"]) // int(sampling["tubelet_size"])
    arrays = {
        side: {"theta": [], "distance": [], "valid": [], "proximal": [], "distal": []}
        for side in SIDES
    }
    take_uids, window_keys = [], []
    geometry_by_take = {}
    for row_index, key in enumerate(keys):
        take_uid, window_index = _parse_window_key(key)
        take = manifest["take_index"].get(take_uid)
        if take is None:
            raise KeyError(f"Split key has no manifest take: {key}")
        if take_uid not in geometry_by_take:
            geometry_by_take[take_uid] = load_take_arm_geometry(
                str(Path(take["pose_path"]).expanduser().resolve()),
                float(pose.get("fps", 30.0)),
                int(pose.get("min_views", 2)),
                int(pose.get("smooth_window", 5)),
            )
        sample_times = build_sample_times(
            float(window_index) * float(config["data"].get("slice_stride_seconds", 12.8)),
            int(sampling["num_frames"]),
            float(sampling["fps"]),
        )
        pooled = pool_window_arm_targets(
            geometry_by_take[take_uid],
            sample_times,
            tubelet_size=int(sampling["tubelet_size"]),
            max_time_delta=float(sampling.get("max_time_delta", 0.05)),
        )
        for side in SIDES:
            if len(pooled[side]["theta"]) != tokens:
                raise RuntimeError(f"Unexpected token count for {key}/{side}")
            arrays[side]["theta"].append(pooled[side]["theta"])
            arrays[side]["distance"].append(pooled[side]["distance"])
            arrays[side]["valid"].append(pooled[side]["valid"])
            arrays[side]["proximal"].append(
                float(pooled[side]["median_proximal_length_m"])
            )
            arrays[side]["distal"].append(
                float(pooled[side]["median_distal_length_m"])
            )
        take_uids.append(take_uid)
        window_keys.append(key)
        progress_every = int(config.get("runtime", {}).get("progress_every", 100))
        if progress_every > 0 and (row_index + 1) % progress_every == 0:
            print(f"{split_name}: {row_index + 1}/{len(keys)} windows", flush=True)
    for side in SIDES:
        for field in ("theta", "distance", "valid"):
            arrays[side][field] = np.stack(arrays[side][field], axis=0)
        for field in ("proximal", "distal"):
            arrays[side][field] = np.asarray(arrays[side][field], dtype=np.float64)
    return {
        "split": split_name,
        "take_uid": np.asarray(take_uids, dtype=str),
        "window_key": np.asarray(window_keys, dtype=str),
        "arms": arrays,
        "unique_take_count": len(geometry_by_take),
    }


def _summarize_side(
    side_data: dict,
    other_data: dict,
    active_threshold: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    theta = side_data["theta"]
    distance = side_data["distance"]
    valid = side_data["valid"].astype(bool)
    proximal = side_data["proximal"][:, None]
    distal = side_data["distal"][:, None]
    correct_prediction = cosine_law_distance(theta, proximal, distal)
    supplementary_prediction = cosine_law_distance(np.pi - theta, proximal, distal)
    wrong_side_prediction = cosine_law_distance(
        other_data["theta"], proximal, distal
    )
    wrong_side_valid = valid & other_data["valid"].astype(bool)
    shifted_theta = np.full_like(theta, np.nan)
    shifted_valid = np.zeros_like(valid)
    shifted_theta[:, :-1] = theta[:, 1:]
    shifted_valid[:, :-1] = valid[:, 1:]
    shifted_prediction = cosine_law_distance(shifted_theta, proximal, distal)
    correct_error = np.where(valid, np.abs(correct_prediction - distance), np.nan)
    supplementary_error = np.where(
        valid, np.abs(supplementary_prediction - distance), np.nan
    )
    wrong_side_error = np.where(
        wrong_side_valid, np.abs(wrong_side_prediction - distance), np.nan
    )
    shifted_error = np.where(
        valid & shifted_valid, np.abs(shifted_prediction - distance), np.nan
    )
    active = _active_mask(theta, valid, active_threshold)
    summary = {
        "valid_tokens": int(np.sum(valid)),
        "total_tokens": int(valid.size),
        "valid_rate": float(np.mean(valid)),
        "oracle_floor": error_summary(correct_prediction, distance, valid),
        "wrong_supplementary": error_summary(
            supplementary_prediction, distance, valid
        ),
        "correct_vs_wrong_supplementary": binary_discrimination(
            correct_error, supplementary_error
        ),
        "correct_vs_wrong_side": binary_discrimination(correct_error, wrong_side_error),
        "correct_vs_one_token_shift": binary_discrimination(correct_error, shifted_error),
        "active_threshold_token_delta_rad": float(active_threshold),
        "active_token_count": int(np.sum(active)),
        "active_oracle_floor": error_summary(correct_prediction, distance, active),
    }
    return summary, {
        "correct_error": correct_error,
        "supplementary_error": supplementary_error,
        "wrong_side_error": wrong_side_error,
        "shifted_error": shifted_error,
        "active": active,
    }


def _per_take_rows(split_data: dict, summaries: dict[str, dict]) -> list[dict]:
    rows = []
    takes = split_data["take_uid"]
    for take_uid in sorted(set(takes.tolist())):
        row_mask = takes == take_uid
        for side in SIDES:
            errors = summaries[side]
            correct = errors["correct_error"][row_mask]
            wrong = errors["supplementary_error"][row_mask]
            valid = np.isfinite(correct) & np.isfinite(wrong)
            rows.append(
                {
                    "split": split_data["split"],
                    "take_uid": take_uid,
                    "side": side,
                    "window_count": int(np.sum(row_mask)),
                    "valid_tokens": int(np.sum(valid)),
                    "oracle_mae_m": float(np.nanmean(correct)) if np.any(valid) else np.nan,
                    "wrong_supplementary_mae_m": (
                        float(np.nanmean(wrong)) if np.any(np.isfinite(wrong)) else np.nan
                    ),
                    "paired_correct_preference": (
                        float(np.mean(wrong[valid] > correct[valid]))
                        if np.any(valid)
                        else np.nan
                    ),
                }
            )
    return rows


def run_gate(config: dict, config_path: Path | None = None) -> dict:
    selection_name = str(config.get("selection_split", "probe_train"))
    validation_name = str(config.get("validation_split", "probe_val"))
    if selection_name != "probe_train" or validation_name != "probe_val":
        raise ValueError("Formal EgoExo gate is frozen to probe_train -> probe_val")
    selection_keys, selection_path = _load_split(config, selection_name)
    validation_keys, validation_path = _load_split(config, validation_name)
    selection_takes = {_parse_window_key(key)[0] for key in selection_keys}
    validation_takes = {_parse_window_key(key)[0] for key in validation_keys}
    if selection_takes & validation_takes:
        raise RuntimeError("probe_train/probe_val take overlap")
    manifest, manifest_path = _load_manifest(config)
    selection = _collect_split(config, manifest, selection_name, selection_keys)
    validation = _collect_split(config, manifest, validation_name, validation_keys)

    active_quantile = float(config.get("gate", {}).get("active_quantile", 0.75))
    active_thresholds = {
        side: _active_threshold(
            selection["arms"][side]["theta"],
            selection["arms"][side]["valid"],
            active_quantile,
        )
        for side in SIDES
    }
    split_summaries, split_errors = {}, {}
    for split_data in (selection, validation):
        arm_summaries, arm_errors = {}, {}
        for side in SIDES:
            arm_summaries[side], arm_errors[side] = _summarize_side(
                split_data["arms"][side],
                split_data["arms"]["right" if side == "left" else "left"],
                active_thresholds[side],
            )
        split_summaries[split_data["split"]] = {
            "window_count": int(len(split_data["window_key"])),
            "take_count": int(split_data["unique_take_count"]),
            "arms": arm_summaries,
        }
        split_errors[split_data["split"]] = arm_errors

    # Store a train-only morphology sensitivity parameter.  The primary model
    # score still uses each take's fixed median morphology.
    global_morphology = {}
    for side in SIDES:
        take_to_lengths = {}
        for take_uid, proximal, distal in zip(
            selection["take_uid"],
            selection["arms"][side]["proximal"],
            selection["arms"][side]["distal"],
        ):
            take_to_lengths.setdefault(str(take_uid), (float(proximal), float(distal)))
        values = np.asarray(list(take_to_lengths.values()), dtype=np.float64)
        values = values[np.all(np.isfinite(values), axis=1)]
        if not len(values):
            raise RuntimeError(
                f"No finite probe-train morphology values for {side}"
            )
        global_morphology[side] = {
            "proximal_length_m": float(np.median(values[:, 0])),
            "distal_length_m": float(np.median(values[:, 1])),
            "take_count": int(len(values)),
        }

    thresholds = config.get("gate", {})
    failures = []
    for split_name, split_summary in split_summaries.items():
        for side, summary in split_summary["arms"].items():
            if summary["valid_rate"] < float(thresholds.get("min_valid_rate", 0.5)):
                failures.append(f"{split_name}/{side}: valid_rate too low")
            if summary["oracle_floor"]["mae_over_mean_target"] > float(
                thresholds.get("max_oracle_relative_mae", 0.10)
            ):
                failures.append(f"{split_name}/{side}: oracle relative MAE too high")
            discrimination = summary["correct_vs_wrong_supplementary"]
            if discrimination["auroc"] < float(
                thresholds.get("min_supplementary_auroc", 0.80)
            ):
                failures.append(f"{split_name}/{side}: supplementary AUROC too low")
            if discrimination["mean_margin_negative_minus_correct_m"] < float(
                thresholds.get("min_supplementary_margin_m", 0.05)
            ):
                failures.append(f"{split_name}/{side}: supplementary margin too low")

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    per_take_paths = {}
    for split_data in (selection, validation):
        path = output_dir / f"{split_data['split']}_per_take.csv"
        _write_csv(
            path,
            _per_take_rows(split_data, split_errors[split_data["split"]]),
        )
        per_take_paths[split_data["split"]] = str(path)
    result = {
        "schema": 1,
        "status": "PASS" if not failures else "FAIL",
        "protocol": GT_GATE_PROTOCOL,
        "selection_split": selection_name,
        "validation_split": validation_name,
        "formula": {
            "name": FORMULA_NAME,
            "expression": FORMULA_EXPRESSION,
            "theta_domain_rad": [0.0, float(np.pi)],
            "morphology": MORPHOLOGY_DESCRIPTION,
            "prediction_angle_clipping": True,
        },
        "active_quantile": active_quantile,
        "active_thresholds": active_thresholds,
        "global_probe_train_morphology": global_morphology,
        "splits": {
            selection_name: {"path": str(selection_path), "sha256": sha256_file(selection_path)},
            validation_name: {"path": str(validation_path), "sha256": sha256_file(validation_path)},
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "config": (
            {"path": str(config_path), "sha256": sha256_file(config_path)}
            if config_path is not None
            else None
        ),
        "thresholds": {
            key: value
            for key, value in thresholds.items()
            if key != "active_quantile"
        },
        "results": split_summaries,
        "per_take_csv": per_take_paths,
        "failures": failures,
        "interpretation_guard": INTERPRETATION_GUARD,
    }
    metrics_path = output_dir / "gt_gate_metrics.json"
    atomic_json_dump(json_safe(result), metrics_path)
    result["metrics_path"] = str(metrics_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, path = _load_yaml(args.config)
    result = run_gate(config, path)
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
