#!/usr/bin/env python3
"""Freeze a train-only GT consistency gate and delta-v normalization scale."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from decord import VideoReader, cpu

from src.time_jepa.covla_law import (
    MAX_DT_SECONDS, PROTOCOL, load_law_series, sha256_file, tubelet_law_metadata,
)
from src.time_jepa.physical_targets import read_covla_rows, sample_clip_indices_from_vr

# Engineering admission criteria, fixed before running this gate. These are
# not statistical claims or downstream model-selection criteria.
CRITERIA = {
    "min_valid_fraction": 0.95,
    "min_correlation": 0.5,
    "min_slope": 0.5,
    "max_slope": 1.5,
    "max_residual_to_delta_v_mae": 0.75,
    "min_timestamp_duration_ratio": 0.95,
    "max_timestamp_duration_ratio": 1.05,
}


def measure_video(item):
    index, row, seed = item
    vr = VideoReader(row["video_path"], num_threads=1, ctx=cpu(0))
    indices = sample_clip_indices_from_vr(
        vr, frames_per_clip=64, fps=5, strict_time_sampling=True,
        rng=np.random.RandomState(seed + index),
    )[0]
    series = load_law_series(row["states_path"])
    video_times = np.asarray(vr.get_frame_timestamp(range(len(vr))), dtype=np.float64)[:, 0]
    if not np.isfinite(video_times).all() or not (np.diff(video_times) > 0).all():
        raise ValueError(f"Nonmonotonic video PTS: {row['video_id']}")
    meta = tubelet_law_metadata(indices, len(vr), series, video_frame_times=video_times)
    target = meta["law_target"].numpy().astype(np.float64)
    valid = meta["law_valid"].numpy()
    dt = meta["law_dt"].numpy().astype(np.float64)
    dv = np.diff(target[:, 0])
    accel = (target[1:, 1] + target[:-1, 1]) / 2
    timestamps = series[1]
    video_duration = (len(vr) - 1) / vr.get_avg_fps()
    duration_ratio = (video_times[-1] - video_times[0]) / video_duration
    raw_duration_ratio = (timestamps[-1] - timestamps[0]) / video_duration
    lags = {}
    for lag in (-2, -1, 0, 1, 2):
        left = np.arange(len(dv))
        right = left + lag
        keep = (right >= 0) & (right < len(dv))
        left, right = left[keep], right[keep]
        keep = valid[left] & valid[right]
        left, right = left[keep], right[keep]
        lags[str(lag)] = (dv[left] / dt[left] - accel[right]).tolist()
    return {
        "dv": dv[valid].tolist(), "dt": dt[valid].tolist(),
        "accel": accel[valid].tolist(), "euler_accel": target[:-1, 1][valid].tolist(),
        "pairs": len(valid), "duration_ratio": float(duration_ratio),
        "raw_state_duration_ratio": float(raw_duration_ratio),
        "fps": float(vr.get_avg_fps()), "lags": lags,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    opts = parser.parse_args()
    cfg = yaml.safe_load(Path(opts.config).read_text())
    data = cfg["data"]
    if data["dataset_fpcs"] != [64] or data["fps"] != 5 or not data["strict_time_sampling"] or data["tubelet_size"] != 2:
        raise ValueError("Gate requires the frozen strict 64f@5FPS, tubelet=2 protocol")
    split = (ROOT / data["encoder_train_split"]).resolve()
    ids = json.loads(split.read_text())
    if len(ids) != len(set(ids)) or len(ids) != 6305:
        raise ValueError("Expected exactly 6305 unique encoder-train video IDs")
    rows_by_id = {row["video_id"]: row for row in read_covla_rows(data["datasets"])}
    rows = [rows_by_id[vid] for vid in sorted(ids)]
    seed = int(cfg["meta"]["seed"])
    with ProcessPoolExecutor(max_workers=opts.workers) as pool:
        records = []
        for i, result in enumerate(pool.map(measure_video, [(i, r, seed) for i, r in enumerate(rows)], chunksize=8)):
            records.append(result)
            if (i + 1) % 250 == 0:
                print(f"GT gate: {i + 1}/{len(rows)} train videos", flush=True)
    def concat(key):
        return np.concatenate([np.asarray(record[key], dtype=np.float64) for record in records])
    dv, dt, accel, euler = [concat(key) for key in ("dv", "dt", "accel", "euler_accel")]
    if len(dv) < 2 or np.std(dv) <= 0:
        raise RuntimeError("Insufficient valid, nonconstant training dynamics")
    derivative = dv / dt
    slope, intercept = np.polyfit(accel, derivative, 1)
    correlation = np.corrcoef(derivative, accel)[0, 1]
    residual = dv - dt * accel
    relative_mae = np.abs(residual).mean() / max(np.abs(dv).mean(), 1e-12)
    ratio = np.asarray([record["duration_ratio"] for record in records])
    valid_fraction = len(dv) / sum(record["pairs"] for record in records)
    checks = {
        "valid_fraction": valid_fraction >= CRITERIA["min_valid_fraction"],
        "correlation": correlation >= CRITERIA["min_correlation"],
        "slope": CRITERIA["min_slope"] <= slope <= CRITERIA["max_slope"],
        "residual_relative_to_motion": relative_mae <= CRITERIA["max_residual_to_delta_v_mae"],
        "timestamp_units": bool(np.isfinite(ratio).all() and ratio.min() >= CRITERIA["min_timestamp_duration_ratio"] and ratio.max() <= CRITERIA["max_timestamp_duration_ratio"]),
    }
    result = {
        "protocol": PROTOCOL, "passed": all(checks.values()),
        "checks": {k: bool(v) for k, v in checks.items()}, "criteria": CRITERIA,
        "split_path": str(split), "split_sha256": sha256_file(split),
        "state_stats_sha256": sha256_file(cfg["time_jepa"]["stats_cache"]),
        "config_sha256": sha256_file(opts.config),
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "src/time_jepa/covla_law.py", "covla_law_rjob/gate.py", "src/time_jepa/physical_targets.py")},
        "video_count": len(rows), "seed": seed,
        "sampling": "sorted train IDs; one exact strict clip/video; RandomState(seed + index)",
        "timestamp_source": "decoded video frame PTS at mapped STATE frame IDs",
        "timestamp_unit": "seconds", "max_dt_seconds": MAX_DT_SECONDS,
        "valid_pairs": len(dv), "valid_fraction": valid_fraction,
        "delta_v_scale_mps": max(float(np.std(dv, ddof=1)), 1e-6),
        "delta_v_scale_ddof": 1, "derivative_acceleration_pearson": float(correlation),
        "derivative_on_acceleration_slope": float(slope), "intercept_mps2": float(intercept),
        "trapezoid_residual_mae_mps": float(np.abs(residual).mean()),
        "euler_residual_mae_mps": float(np.abs(dv - dt * euler).mean()),
        "delta_v_mae_mps": float(np.abs(dv).mean()), "residual_to_delta_v_mae": float(relative_mae),
        "dt_seconds_quantiles": np.quantile(dt, [0, .01, .5, .99, 1]).tolist(),
        "timestamp_duration_ratio_min_max": [float(ratio.min()), float(ratio.max())],
        "raw_state_timestamp_bad_duration_videos": sum(
            not .95 <= r["raw_state_duration_ratio"] <= 1.05 for r in records),
        "raw_state_timestamp_duration_ratio_min_max": [
            min(r["raw_state_duration_ratio"] for r in records),
            max(r["raw_state_duration_ratio"] for r in records)],
        "source_fps_min_max": [min(r["fps"] for r in records), max(r["fps"] for r in records)],
        "lag_derivative_residual_mae_mps2": {
            str(lag): float(np.abs(np.concatenate([r["lags"][str(lag)] for r in records])).mean())
            for lag in (-2, -1, 0, 1, 2)
        },
    }
    output = Path(opts.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("GT consistency gate failed; do not launch training")


if __name__ == "__main__":
    main()
