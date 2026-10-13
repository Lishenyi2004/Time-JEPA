#!/usr/bin/env python3
"""Evaluate independently decoded COVLA variables under the frozen GT law.

The accuracy-matched null keeps every head's complete temporal error trace but
reassigns those traces independently between videos inside frozen motion
strata.  It therefore preserves per-head error scale and temporal correlation
while destroying cross-head alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

try:
    from .bank_common import (
        PRIMARY_HORIZONS,
        TARGET_INDEX,
        TARGET_NAMES,
        VA_HORIZONS,
        atomic_json_dump,
        json_safe,
        sha256_file,
    )
    from .covla_gt_consistency_gate import (
        integrate_scalar,
        horizon_layout,
        mean_absolute_over_window,
        rollout_uniform,
        value_at_horizon,
    )
except ImportError:
    from bank_common import (
        PRIMARY_HORIZONS,
        TARGET_INDEX,
        TARGET_NAMES,
        VA_HORIZONS,
        atomic_json_dump,
        json_safe,
        sha256_file,
    )
    from covla_gt_consistency_gate import (
        integrate_scalar,
        horizon_layout,
        mean_absolute_over_window,
        rollout_uniform,
        value_at_horizon,
    )


NEGATIVE_NAMES = (
    "wrong_yaw_sign",
    "wrong_vy_sign",
    "one_token_time_shift",
    "matched_wrong_video",
    "wrong_horizon",
)
JST = timezone(timedelta(hours=9))


def _capture_date(covla_root: str | Path, video_id: str) -> str:
    state_path = Path(covla_root) / "states" / f"{video_id}.jsonl"
    try:
        with state_path.open() as handle:
            first = json.loads(handle.readline())
        payload = next(iter(first.values()))
        timestamp_ms = int(payload["timestamp"])
        if timestamp_ms <= 0:
            return "unknown"
        return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).astimezone(JST).date().isoformat()
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError):
        return "unknown"


def _finite_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def _ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"low": None, "high": None, "median": None}
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "median": float(np.quantile(values, 0.5)),
    }


def _masked(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64).copy()
    result[~np.asarray(valid, dtype=bool)] = np.nan
    return result


def _point_metrics(prediction, target, valid, train_mean, train_std) -> dict:
    results = {}
    for index, name in enumerate(TARGET_NAMES):
        mask = valid[..., index].astype(bool)
        pred = prediction[..., index][mask].astype(np.float64)
        truth = target[..., index][mask].astype(np.float64)
        finite = np.isfinite(pred) & np.isfinite(truth)
        pred, truth = pred[finite], truth[finite]
        if not len(pred):
            results[name] = {"count": 0}
            continue
        error = pred - truth
        mse = float(np.mean(error**2))
        target_variance_sum = float(np.sum((truth - np.mean(truth)) ** 2))
        baseline_sum = float(np.sum((truth - float(train_mean[index])) ** 2))
        if len(pred) > 1 and np.std(pred) > 0 and np.std(truth) > 0:
            pearson = float(np.corrcoef(pred, truth)[0, 1])
        else:
            pearson = 0.0
        covariance = float(np.mean((pred - np.mean(pred)) * (truth - np.mean(truth))))
        ccc_denom = float(np.var(pred) + np.var(truth) + (np.mean(pred) - np.mean(truth)) ** 2)
        results[name] = {
            "count": int(len(pred)),
            "valid_rate": float(len(pred) / max(1, valid.shape[0] * valid.shape[1])),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(math.sqrt(mse)),
            "nrmse": float(math.sqrt(mse) / max(float(train_std[index]), 1e-6)),
            "pearson": pearson,
            "ccc": float(2 * covariance / ccc_denom) if ccc_denom > 1e-12 else 0.0,
            "r2": float(1 - np.sum(error**2) / target_variance_sum)
            if target_variance_sum > 1e-12
            else 0.0,
            "skill_train_mean": float(1 - np.sum(error**2) / baseline_sum)
            if baseline_sum > 1e-12
            else 0.0,
        }
    return results


def _target_pair(point: int) -> tuple[int, int]:
    return TARGET_INDEX[f"trajectory_{point}_x"], TARGET_INDEX[f"trajectory_{point}_y"]


def _sequence_components(values: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: _masked(values[:, index], valid[:, index])
        for name, index in TARGET_INDEX.items()
    }


def _trajectory(values: np.ndarray, valid: np.ndarray, point: int) -> np.ndarray:
    x_index, y_index = _target_pair(point)
    result = np.asarray(values[:, [x_index, y_index]], dtype=np.float64).copy()
    mask = valid[:, [x_index, y_index]].astype(bool).all(axis=1)
    result[~mask] = np.nan
    return result


def _rollout(components: dict[str, np.ndarray], dt: float, horizon: float, **signs) -> np.ndarray:
    return rollout_uniform(
        components["vx_calib"],
        components["vy_calib"],
        components["yaw_rate"],
        dt,
        horizon,
        "trapezoid",
        vy_sign=int(signs.get("vy_sign", -1)),
        yaw_sign=int(signs.get("yaw_sign", -1)),
        rotate=True,
    )


def _residual_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    target_frame_indices: np.ndarray,
    video_fps: float,
    thresholds: dict,
) -> dict:
    pred_components = _sequence_components(prediction, valid)
    gt_components = _sequence_components(target, valid)
    centers = np.mean(target_frame_indices, axis=1)
    dt = float(np.median(np.diff(centers)) / video_fps)
    result = {"dt": dt, "horizons": {}, "va": {}}
    for horizon, point in PRIMARY_HORIZONS:
        pred_rollout = _rollout(pred_components, dt, horizon)
        gt_rollout = _rollout(gt_components, dt, horizon)
        pred_trajectory = _trajectory(prediction, valid, point)[: len(pred_rollout)]
        gt_trajectory = _trajectory(target, valid, point)[: len(pred_rollout)]
        count = min(len(pred_rollout), len(gt_rollout), len(pred_trajectory), len(gt_trajectory))
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
        raw_xy = pred_trajectory - pred_rollout
        oracle_xy = gt_trajectory - gt_rollout
        excess_xy = raw_xy - oracle_xy
        raw_norm = np.linalg.norm(raw_xy, axis=1)
        oracle_norm = np.linalg.norm(oracle_xy, axis=1)
        excess_norm = np.linalg.norm(excess_xy, axis=1)
        target_norm = np.linalg.norm(gt_trajectory, axis=1)
        turn = mean_absolute_over_window(gt_components["yaw_rate"], dt, 1.0)[:count]
        acceleration = mean_absolute_over_window(gt_components["a_ego"], dt, 1.0)[:count]
        signed_acceleration = (
            integrate_scalar(gt_components["a_ego"], dt, 1.0, "trapezoid")[:count]
        )
        speed = mean_absolute_over_window(gt_components["v_ego"], dt, 1.0)[:count]
        acceleration_threshold = thresholds["active_acceleration_q75_m_per_s2"]
        lateral_threshold = thresholds["active_lateral_q75_m_by_horizon"][str(horizon)]
        condition_masks = {
            "overall": np.ones(count, dtype=bool),
            "active_turning": turn >= thresholds["active_turning_q75_rad_per_s"],
            "straight": turn < thresholds["active_turning_q75_rad_per_s"],
            "active_acceleration": acceleration
            >= acceleration_threshold,
            "accelerating": signed_acceleration >= acceleration_threshold,
            "braking": signed_acceleration <= -acceleration_threshold,
            "low_speed": speed <= thresholds["low_speed_q10_m_per_s"],
            "high_speed": speed >= thresholds["high_speed_q75_m_per_s"],
            "active_lateral": np.abs(gt_trajectory[:, 1]) >= lateral_threshold,
        }
        result["horizons"][str(horizon)] = {
            "finite": finite,
            "raw_xy": raw_xy,
            "oracle_xy": oracle_xy,
            "excess_xy": excess_xy,
            "raw_norm": raw_norm,
            "oracle_norm": oracle_norm,
            "excess_norm": excess_norm,
            "target_norm": target_norm,
            "condition_masks": condition_masks,
            "pred_rollout": pred_rollout,
            "gt_rollout": gt_rollout,
            "pred_trajectory": pred_trajectory,
            "gt_trajectory": gt_trajectory,
        }
    for horizon in VA_HORIZONS:
        pred_integral = integrate_scalar(pred_components["a_ego"], dt, horizon, "trapezoid")
        gt_integral = integrate_scalar(gt_components["a_ego"], dt, horizon, "trapezoid")
        pred_future = value_at_horizon(pred_components["v_ego"], dt, horizon)
        gt_future = value_at_horizon(gt_components["v_ego"], dt, horizon)
        count = min(len(pred_integral), len(gt_integral), len(pred_future), len(gt_future))
        pred_residual = pred_future[:count] - pred_components["v_ego"][:count] - pred_integral[:count]
        gt_residual = gt_future[:count] - gt_components["v_ego"][:count] - gt_integral[:count]
        finite = np.isfinite(pred_residual) & np.isfinite(gt_residual)
        result["va"][str(horizon)] = {
            "finite": finite,
            "raw": pred_residual,
            "oracle": gt_residual,
        }
    return result


def _aggregate_residuals(per_video: list[dict]) -> tuple[dict, list[dict]]:
    aggregate = {}
    csv_rows = []
    for horizon, _ in PRIMARY_HORIZONS:
        key = str(horizon)
        condition_result = {}
        for condition in (
            "overall",
            "active_turning",
            "straight",
            "active_acceleration",
            "accelerating",
            "braking",
            "low_speed",
            "high_speed",
            "active_lateral",
        ):
            raw_chunks, oracle_chunks, excess_chunks, target_chunks = [], [], [], []
            x_chunks, y_chunks = [], []
            potential = 0
            for item in per_video:
                record = item["residuals"]["horizons"][key]
                condition_mask = record["condition_masks"][condition]
                potential += int(np.sum(condition_mask))
                mask = record["finite"] & condition_mask
                raw_chunks.append(record["raw_norm"][mask])
                oracle_chunks.append(record["oracle_norm"][mask])
                excess_chunks.append(record["excess_norm"][mask])
                target_chunks.append(record["target_norm"][mask])
                x_chunks.append(np.abs(record["raw_xy"][mask, 0]))
                y_chunks.append(np.abs(record["raw_xy"][mask, 1]))
            raw = np.concatenate(raw_chunks) if raw_chunks else np.empty(0)
            oracle = np.concatenate(oracle_chunks) if oracle_chunks else np.empty(0)
            excess = np.concatenate(excess_chunks) if excess_chunks else np.empty(0)
            target = np.concatenate(target_chunks) if target_chunks else np.empty(0)
            x_error = np.concatenate(x_chunks) if x_chunks else np.empty(0)
            y_error = np.concatenate(y_chunks) if y_chunks else np.empty(0)
            condition_result[condition] = {
                "count": int(len(raw)),
                "potential": int(potential),
                "valid_rate": float(len(raw) / max(1, potential)),
                "raw_2d_mae_m": float(np.mean(raw)) if len(raw) else None,
                "raw_x_mae_m": float(np.mean(x_error)) if len(raw) else None,
                "raw_y_mae_m": float(np.mean(y_error)) if len(raw) else None,
                "oracle_2d_mae_m": float(np.mean(oracle)) if len(raw) else None,
                "model_excess_2d_mae_m": float(np.mean(excess)) if len(raw) else None,
                "target_mean_norm_m": float(np.mean(target)) if len(raw) else None,
                "raw_relative_2d_mae": float(np.mean(raw) / max(np.mean(target), 1e-12))
                if len(raw)
                else None,
            }
        aggregate[key] = {"conditions": condition_result}
        for item in per_video:
            record = item["residuals"]["horizons"][key]
            mask = record["finite"]
            raw = record["raw_norm"][mask]
            oracle = record["oracle_norm"][mask]
            excess = record["excess_norm"][mask]
            target = record["target_norm"][mask]
            csv_rows.append(
                {
                    "video_id": item["video_id"],
                    "horizon_seconds": horizon,
                    "count": int(len(raw)),
                    "raw_sum_m": float(np.sum(raw)),
                    "raw_2d_mae_m": float(np.mean(raw)) if len(raw) else math.nan,
                    "oracle_sum_m": float(np.sum(oracle)),
                    "oracle_2d_mae_m": float(np.mean(oracle)) if len(raw) else math.nan,
                    "excess_sum_m": float(np.sum(excess)),
                    "excess_2d_mae_m": float(np.mean(excess)) if len(raw) else math.nan,
                    "target_norm_sum_m": float(np.sum(target)),
                    "target_mean_norm_m": float(np.mean(target)) if len(raw) else math.nan,
                }
            )
    return aggregate, csv_rows


def _aggregate_va(per_video: list[dict]) -> dict:
    output = {}
    for horizon in VA_HORIZONS:
        raw_chunks, oracle_chunks = [], []
        for item in per_video:
            record = item["residuals"]["va"][str(horizon)]
            raw_chunks.append(record["raw"][record["finite"]])
            oracle_chunks.append(record["oracle"][record["finite"]])
        raw = np.concatenate(raw_chunks) if raw_chunks else np.empty(0)
        oracle = np.concatenate(oracle_chunks) if oracle_chunks else np.empty(0)
        output[str(horizon)] = {
            "count": int(len(raw)),
            "raw_mae_m_per_s": float(np.mean(np.abs(raw))) if len(raw) else None,
            "oracle_mae_m_per_s": float(np.mean(np.abs(oracle))) if len(raw) else None,
            "model_excess_mae_m_per_s": float(np.mean(np.abs(raw - oracle)))
            if len(raw)
            else None,
        }
    return output


def _motion_strata(target: np.ndarray, valid: np.ndarray, target_frames, video_fps, thresholds):
    labels = []
    strengths = []
    for values, mask, frames in zip(target, valid, target_frames):
        components = _sequence_components(values, mask)
        dt = float(np.median(np.diff(np.mean(frames, axis=1))) / video_fps)
        turn = float(np.nanmean(mean_absolute_over_window(components["yaw_rate"], dt, 1.0)))
        acceleration = float(np.nanmean(mean_absolute_over_window(components["a_ego"], dt, 1.0)))
        speed = float(np.nanmean(mean_absolute_over_window(components["v_ego"], dt, 1.0)))
        if speed <= thresholds["low_speed_q10_m_per_s"]:
            speed_bin = "low"
        elif speed >= thresholds["high_speed_q75_m_per_s"]:
            speed_bin = "high"
        else:
            speed_bin = "mid"
        label = f"speed={speed_bin}|turn={int(turn >= thresholds['active_turning_q75_rad_per_s'])}|accel={int(acceleration >= thresholds['active_acceleration_q75_m_per_s2'])}"
        labels.append(label)
        strengths.append({"speed": speed, "turn": turn, "acceleration": acceleration})
    return np.asarray(labels), strengths


def _effective_strata(labels: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.asarray(labels, dtype=str)
    counts = {label: int(np.sum(labels == label)) for label in set(labels.tolist())}
    effective = labels.copy()
    fallback = 0
    for index, label in enumerate(labels):
        if counts[label] < 2:
            effective[index] = f"{label.split('|')[0]}|coarse"
            fallback += 1
    coarse_counts = {
        label: int(np.sum(effective == label)) for label in set(effective.tolist())
    }
    for index, label in enumerate(effective.copy()):
        if coarse_counts[label] < 2:
            effective[index] = "global"
    return effective, fallback


def _donor_permutation(
    labels: np.ndarray, rng: np.random.Generator, derange: bool = False
) -> tuple[np.ndarray, int]:
    count = len(labels)
    donors = np.arange(count)
    effective, fallback = _effective_strata(labels)
    groups = defaultdict(list)
    for index, label in enumerate(effective.tolist()):
        groups[label].append(index)
    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) > 1:
            if derange:
                shuffled = rng.permutation(indices)
                donors[shuffled] = np.roll(shuffled, 1)
            else:
                # Fixed points are allowed in a permutation test. This is
                # essential for two-video strata: forcing a derangement would
                # give every scalar head the same swap and preserve alignment.
                donors[indices] = rng.permutation(indices)
    return donors, fallback


def _pooled_horizon_scores(per_video: list[dict]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    raw_residual, excess_residual, scale = {}, {}, {}
    for horizon, _ in PRIMARY_HORIZONS:
        key = str(horizon)
        raw, excess, target = [], [], []
        for item in per_video:
            record = item["residuals"]["horizons"][key]
            mask = record["finite"]
            raw.append(record["raw_norm"][mask])
            excess.append(record["excess_norm"][mask])
            target.append(record["target_norm"][mask])
        raw = np.concatenate(raw)
        excess = np.concatenate(excess)
        target = np.concatenate(target)
        raw_residual[key] = float(np.mean(raw))
        excess_residual[key] = float(np.mean(excess))
        scale[key] = float(np.mean(target))
    return raw_residual, excess_residual, scale


def _pooled_va_excess(per_video: list[dict]) -> dict[str, float]:
    result = {}
    for horizon in VA_HORIZONS:
        chunks = []
        for item in per_video:
            record = item["residuals"]["va"][str(horizon)]
            chunks.append(np.abs(record["raw"][record["finite"]] - record["oracle"][record["finite"]]))
        values = np.concatenate(chunks)
        result[str(horizon)] = float(np.mean(values))
    return result


def _rollout_batch(
    vx: np.ndarray,
    vy: np.ndarray,
    omega: np.ndarray,
    dt: float,
    horizon: float,
    vy_sign: int = -1,
    yaw_sign: int = -1,
) -> np.ndarray:
    """Vectorized equivalent of the frozen trapezoid SE(2) rollout."""
    full, fraction, required = horizon_layout(horizon, dt)
    count = vx.shape[1] - required
    if count <= 0:
        return np.empty((vx.shape[0], 0, 2), dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64) * float(vy_sign)
    omega = np.asarray(omega, dtype=np.float64) * float(yaw_sign)
    pos_x = np.zeros((vx.shape[0], count), dtype=np.float64)
    pos_y = np.zeros_like(pos_x)
    phi = np.zeros_like(pos_x)

    def rotate(x_component, y_component, angle):
        cosine, sine = np.cos(angle), np.sin(angle)
        return (
            cosine * x_component - sine * y_component,
            sine * x_component + cosine * y_component,
        )

    for step in range(full):
        x0, y0 = vx[:, step : step + count], vy[:, step : step + count]
        x1 = vx[:, step + 1 : step + 1 + count]
        y1 = vy[:, step + 1 : step + 1 + count]
        omega0 = omega[:, step : step + count]
        omega1 = omega[:, step + 1 : step + 1 + count]
        phi_next = phi + 0.5 * (omega0 + omega1) * dt
        rx0, ry0 = rotate(x0, y0, phi)
        rx1, ry1 = rotate(x1, y1, phi_next)
        pos_x += 0.5 * (rx0 + rx1) * dt
        pos_y += 0.5 * (ry0 + ry1) * dt
        phi = phi_next
    if fraction > 0.0:
        step = full
        partial_dt = fraction * dt
        x0, y0 = vx[:, step : step + count], vy[:, step : step + count]
        x_follow = vx[:, step + 1 : step + 1 + count]
        y_follow = vy[:, step + 1 : step + 1 + count]
        omega0 = omega[:, step : step + count]
        omega_follow = omega[:, step + 1 : step + 1 + count]
        x_end = x0 + fraction * (x_follow - x0)
        y_end = y0 + fraction * (y_follow - y0)
        omega_end = omega0 + fraction * (omega_follow - omega0)
        phi_end = phi + 0.5 * (omega0 + omega_end) * partial_dt
        rx0, ry0 = rotate(x0, y0, phi)
        rx1, ry1 = rotate(x_end, y_end, phi_end)
        pos_x += 0.5 * (rx0 + rx1) * partial_dt
        pos_y += 0.5 * (ry0 + ry1) * partial_dt
    return np.stack([pos_x, pos_y], axis=-1)


def _integrate_scalar_batch(values: np.ndarray, dt: float, horizon: float) -> np.ndarray:
    full, fraction, required = horizon_layout(horizon, dt)
    count = values.shape[1] - required
    output = np.zeros((values.shape[0], max(count, 0)), dtype=np.float64)
    for step in range(full):
        output += 0.5 * (
            values[:, step : step + count] + values[:, step + 1 : step + 1 + count]
        ) * dt
    if fraction > 0 and count > 0:
        step = full
        partial_dt = fraction * dt
        current = values[:, step : step + count]
        following = values[:, step + 1 : step + 1 + count]
        endpoint = current + fraction * (following - current)
        output += 0.5 * (current + endpoint) * partial_dt
    return output


def _value_at_horizon_batch(values: np.ndarray, dt: float, horizon: float) -> np.ndarray:
    full, fraction, required = horizon_layout(horizon, dt)
    count = values.shape[1] - required
    if fraction == 0:
        return values[:, full : full + count]
    current = values[:, full : full + count]
    following = values[:, full + 1 : full + 1 + count]
    return current + fraction * (following - current)


def _batched_excess_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    target_frames: np.ndarray,
    video_fps: float,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, float]]:
    centers = np.mean(target_frames, axis=2)
    dts = np.median(np.diff(centers, axis=1), axis=1) / float(video_fps)
    if not np.allclose(dts, dts[0], rtol=0.0, atol=1e-9):
        raise RuntimeError("Batched null requires the frozen uniform COVLA time grid")
    dt = float(dts[0])
    prediction = np.asarray(prediction, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64).copy()
    prediction[~valid.astype(bool)] = np.nan
    target[~valid.astype(bool)] = np.nan
    pred_vx = prediction[..., TARGET_INDEX["vx_calib"]]
    pred_vy = prediction[..., TARGET_INDEX["vy_calib"]]
    pred_yaw = prediction[..., TARGET_INDEX["yaw_rate"]]
    gt_vx = target[..., TARGET_INDEX["vx_calib"]]
    gt_vy = target[..., TARGET_INDEX["vy_calib"]]
    gt_yaw = target[..., TARGET_INDEX["yaw_rate"]]
    pooled, per_video = {}, {}
    for horizon, point in PRIMARY_HORIZONS:
        key = str(horizon)
        pred_rollout = _rollout_batch(pred_vx, pred_vy, pred_yaw, dt, horizon)
        gt_rollout = _rollout_batch(gt_vx, gt_vy, gt_yaw, dt, horizon)
        count = pred_rollout.shape[1]
        x_index, y_index = _target_pair(point)
        pred_trajectory = prediction[:, :count, [x_index, y_index]]
        gt_trajectory = target[:, :count, [x_index, y_index]]
        raw = pred_trajectory - pred_rollout
        oracle = gt_trajectory - gt_rollout
        excess = np.linalg.norm(raw - oracle, axis=2)
        finite = np.isfinite(excess)
        counts = finite.sum(axis=1)
        sums = np.where(finite, excess, 0.0).sum(axis=1)
        scores = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan, dtype=np.float64),
            where=counts > 0,
        )
        pooled[key] = float(np.sum(sums) / max(int(np.sum(counts)), 1))
        per_video[key] = scores
    pred_v = prediction[..., TARGET_INDEX["v_ego"]]
    pred_a = prediction[..., TARGET_INDEX["a_ego"]]
    gt_v = target[..., TARGET_INDEX["v_ego"]]
    gt_a = target[..., TARGET_INDEX["a_ego"]]
    va = {}
    for horizon in VA_HORIZONS:
        pred_residual = (
            _value_at_horizon_batch(pred_v, dt, horizon)
            - pred_v[:, : pred_v.shape[1] - horizon_layout(horizon, dt)[2]]
            - _integrate_scalar_batch(pred_a, dt, horizon)
        )
        gt_residual = (
            _value_at_horizon_batch(gt_v, dt, horizon)
            - gt_v[:, : gt_v.shape[1] - horizon_layout(horizon, dt)[2]]
            - _integrate_scalar_batch(gt_a, dt, horizon)
        )
        excess = np.abs(pred_residual - gt_residual)
        finite = np.isfinite(excess)
        va[str(horizon)] = float(
            np.where(finite, excess, 0.0).sum() / max(int(finite.sum()), 1)
        )
    return pooled, per_video, va


def _accuracy_matched_null(
    prediction,
    target,
    valid,
    target_frames,
    video_ids,
    video_fps,
    thresholds,
    permutations,
    seed,
):
    observed_items = [
        {
            "video_id": str(video_ids[i]),
            "residuals": _residual_rows(
                prediction[i], target[i], valid[i], target_frames[i], video_fps, thresholds
            ),
        }
        for i in range(len(video_ids))
    ]
    observed_raw, observed_excess, scales = _pooled_horizon_scores(observed_items)
    observed_va_excess = _pooled_va_excess(observed_items)
    observed_composite = float(
        np.mean([observed_excess[key] / max(scales[key], 1e-12) for key in observed_excess])
    )
    labels, strengths = _motion_strata(target, valid, target_frames, video_fps, thresholds)
    effective_labels, _ = _effective_strata(labels)
    errors = prediction - target
    errors[~valid.astype(bool)] = np.nan
    rng = np.random.default_rng(int(seed))
    null_by_horizon = {str(h): np.empty(permutations, dtype=np.float64) for h, _ in PRIMARY_HORIZONS}
    null_va = {str(h): np.empty(permutations, dtype=np.float64) for h in VA_HORIZONS}
    null_composite = np.empty(permutations, dtype=np.float64)
    video_null_scores = {
        str(h): np.full((permutations, len(video_ids)), np.nan, dtype=np.float64)
        for h, _ in PRIMARY_HORIZONS
    }
    fallback_total = 0
    for permutation in range(permutations):
        null_prediction = np.empty_like(prediction)
        null_valid = np.asarray(valid, dtype=bool).copy()
        for target_index in range(len(TARGET_NAMES)):
            donors, fallback = _donor_permutation(labels, rng)
            fallback_total += fallback
            donor_error = errors[donors, :, target_index]
            donor_valid = np.isfinite(donor_error)
            null_prediction[:, :, target_index] = target[:, :, target_index] + np.nan_to_num(
                donor_error, nan=0.0
            )
            null_valid[:, :, target_index] &= donor_valid
        scores, per_video_scores, va_scores = _batched_excess_scores(
            null_prediction,
            target,
            null_valid,
            target_frames,
            video_fps,
        )
        for key, score in scores.items():
            null_by_horizon[key][permutation] = score
            video_null_scores[key][permutation] = per_video_scores[key]
        for key, score in va_scores.items():
            null_va[key][permutation] = score
        null_composite[permutation] = np.mean(
            [scores[key] / max(scales[key], 1e-12) for key in scores]
        )
    per_horizon = {}
    for key, values in null_by_horizon.items():
        mean_null = float(np.mean(values))
        median_null = float(np.median(values))
        per_horizon[key] = {
            "observed_raw_2d_mae_m": observed_raw[key],
            "observed_model_excess_2d_mae_m": observed_excess[key],
            "null": _finite_summary(values),
            "coherence_gain": float(1.0 - observed_excess[key] / max(median_null, 1e-12)),
            "permutation_p_lower_is_better": float(
                (1 + np.sum(values <= observed_excess[key])) / (len(values) + 1)
            ),
        }
    va_result = {}
    for key, values in null_va.items():
        median_null = float(np.median(values))
        va_result[key] = {
            "observed_model_excess_mae_m_per_s": observed_va_excess[key],
            "null": _finite_summary(values),
            "coherence_gain": float(
                1.0 - observed_va_excess[key] / max(median_null, 1e-12)
            ),
            "permutation_p_lower_is_better": float(
                (1 + np.sum(values <= observed_va_excess[key])) / (len(values) + 1)
            ),
        }
    median_null_composite = float(np.median(null_composite))
    result = {
        "definition": (
            "Each scalar head's complete per-video temporal error trace is independently "
            "reassigned within frozen speed/turn/acceleration strata; recipient GT is retained."
        ),
        "permutations": int(permutations),
        "seed": int(seed),
        "strata_counts": {
            str(label): int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))
        },
        "effective_strata_counts": {
            str(label): int(np.sum(effective_labels == label))
            for label in sorted(set(effective_labels.tolist()))
        },
        "singleton_fallback_assignments": int(fallback_total),
        "horizons": per_horizon,
        "velocity_acceleration_auxiliary": va_result,
        "composite": {
            "definition": "mean over 1.0/2.0/2.95 s of model-excess MAE divided by the same horizon's mean GT path norm",
            "observed": observed_composite,
            "null": _finite_summary(null_composite),
            "S_coh": float(1.0 - observed_composite / max(median_null_composite, 1e-12)),
            "permutation_p_lower_is_better": float(
                (1 + np.sum(null_composite <= observed_composite)) / (len(null_composite) + 1)
            ),
        },
        "motion_strengths": strengths,
        "_null_arrays": null_by_horizon,
        "_null_va_arrays": null_va,
        "_null_composite": null_composite,
        "_video_null_median": {
            key: np.nanmedian(values, axis=0) for key, values in video_null_scores.items()
        },
    }
    return result, observed_items


def _wrong_video_donors(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 991)
    donors, _ = _donor_permutation(labels, rng, derange=True)
    return donors


def _binary_discrimination(correct: np.ndarray, negative: np.ndarray) -> dict:
    correct = np.asarray(correct, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if not len(correct) or not len(negative):
        return {"count": 0}
    sorted_negative = np.sort(negative)
    smaller = np.searchsorted(sorted_negative, correct, side="left")
    no_larger = np.searchsorted(sorted_negative, correct, side="right")
    greater = len(sorted_negative) - no_larger
    equal = no_larger - smaller
    auroc = float(np.sum(greater + 0.5 * equal) / (len(correct) * len(negative)))
    paired_count = min(len(correct), len(negative))
    paired_correct = correct[:paired_count]
    paired_negative = negative[:paired_count]
    return {
        "count": int(len(correct)),
        "auroc": auroc,
        "paired_preference_rate": float(
            np.mean(paired_correct < paired_negative)
            + 0.5 * np.mean(paired_correct == paired_negative)
        ),
        "mean_margin_negative_minus_correct_m": float(
            np.mean(paired_negative - paired_correct)
        ),
    }


def _hard_negative_scores(
    values,
    valid,
    ground_truth,
    ground_valid,
    frames,
    video_fps,
    thresholds,
    labels,
    donors,
):
    all_scores = {str(h): {name: [] for name in ("correct",) + NEGATIVE_NAMES} for h, _ in PRIMARY_HORIZONS}
    all_conditions = {
        str(h): {"active_turning": [], "active_lateral": [], "active_acceleration": []}
        for h, _ in PRIMARY_HORIZONS
    }
    source = values
    for video_index in range(len(values)):
        components = _sequence_components(source[video_index], valid[video_index])
        donor_components = _sequence_components(source[donors[video_index]], valid[donors[video_index]])
        gt_components = _sequence_components(
            ground_truth[video_index], ground_valid[video_index]
        )
        dt = float(np.median(np.diff(np.mean(frames[video_index], axis=1))) / video_fps)
        for horizon_index, (horizon, point) in enumerate(PRIMARY_HORIZONS):
            key = str(horizon)
            target_trajectory = _trajectory(source[video_index], valid[video_index], point)
            candidates = {
                "correct": _rollout(components, dt, horizon),
                "wrong_yaw_sign": _rollout(components, dt, horizon, yaw_sign=1, vy_sign=-1),
                "wrong_vy_sign": _rollout(components, dt, horizon, yaw_sign=-1, vy_sign=1),
                "one_token_time_shift": _rollout(
                    {name: component[1:] for name, component in components.items()}, dt, horizon
                ),
                "matched_wrong_video": _rollout(donor_components, dt, horizon),
            }
            alternate_horizon = PRIMARY_HORIZONS[(horizon_index + 1) % len(PRIMARY_HORIZONS)][0]
            candidates["wrong_horizon"] = _rollout(components, dt, alternate_horizon)
            common = min([len(target_trajectory)] + [len(item) for item in candidates.values()])
            if common <= 0:
                continue
            finite = np.all(np.isfinite(target_trajectory[:common]), axis=1)
            distances = {}
            for name, rollout in candidates.items():
                distances[name] = np.linalg.norm(target_trajectory[:common] - rollout[:common], axis=1)
                finite &= np.isfinite(distances[name])
            for name, distance in distances.items():
                all_scores[key][name].append(distance[finite])
            gt_trajectory = _trajectory(
                ground_truth[video_index], ground_valid[video_index], point
            )[:common]
            turn = mean_absolute_over_window(gt_components["yaw_rate"], dt, 1.0)[:common]
            acceleration = mean_absolute_over_window(gt_components["a_ego"], dt, 1.0)[:common]
            conditions = {
                "active_turning": turn >= thresholds["active_turning_q75_rad_per_s"],
                "active_lateral": np.abs(gt_trajectory[:, 1])
                >= thresholds["active_lateral_q75_m_by_horizon"][key],
                "active_acceleration": acceleration
                >= thresholds["active_acceleration_q75_m_per_s2"],
            }
            for name, condition in conditions.items():
                all_conditions[key][name].append(condition[finite])
    result = {}
    for key, score_chunks in all_scores.items():
        scores = {
            name: np.concatenate(chunks) if chunks else np.empty(0)
            for name, chunks in score_chunks.items()
        }
        correct = scores["correct"]
        if not len(correct):
            result[key] = {"count": 0}
            continue
        negative_matrix = np.stack([scores[name] for name in NEGATIVE_NAMES], axis=1)
        conditions = {
            name: np.concatenate(chunks) if chunks else np.zeros(len(correct), dtype=bool)
            for name, chunks in all_conditions[key].items()
        }
        ranks = 1.0 + np.sum(negative_matrix < correct[:, None], axis=1)
        ranks += 0.5 * np.sum(negative_matrix == correct[:, None], axis=1)
        pairwise = {}
        recommended_condition = {
            "wrong_yaw_sign": "active_turning",
            "wrong_vy_sign": "active_lateral",
            "one_token_time_shift": "active_acceleration",
            "matched_wrong_video": "overall",
            "wrong_horizon": "overall",
        }
        for column, name in enumerate(NEGATIVE_NAMES):
            negative = negative_matrix[:, column]
            condition_name = recommended_condition[name]
            condition = (
                np.ones(len(correct), dtype=bool)
                if condition_name == "overall"
                else conditions[condition_name]
            )
            pairwise[name] = {
                "recommended_condition": condition_name,
                "overall": _binary_discrimination(correct, negative),
                "conditioned": _binary_discrimination(correct[condition], negative[condition]),
            }
        result[key] = {
            "count": int(len(correct)),
            "recall_at_1": float((correct[:, None] <= negative_matrix).all(axis=1).mean()),
            "mrr": float(np.mean(1.0 / ranks)),
            "mean_margin_hardest_negative_minus_correct_m": float(
                np.mean(np.min(negative_matrix, axis=1) - correct)
            ),
            "pairwise": pairwise,
        }
    return result


def _cluster_bootstrap(csv_rows, null_result, video_ids, replicates, seed):
    by_horizon = defaultdict(dict)
    for row in csv_rows:
        by_horizon[str(row["horizon_seconds"])][row["video_id"]] = row
    video_index = {str(video_id): index for index, video_id in enumerate(video_ids)}
    rng = np.random.default_rng(int(seed))
    output = {}
    for horizon, _ in PRIMARY_HORIZONS:
        key = str(horizon)
        ids = [str(video_id) for video_id in video_ids if str(video_id) in by_horizon[key]]
        raw_boot = np.empty(replicates)
        oracle_boot = np.empty(replicates)
        excess_boot = np.empty(replicates)
        coherence_boot = np.empty(replicates)
        for replicate in range(replicates):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            raw_sum = sum(by_horizon[key][item]["raw_sum_m"] for item in sampled)
            oracle_sum = sum(by_horizon[key][item]["oracle_sum_m"] for item in sampled)
            excess_sum = sum(by_horizon[key][item]["excess_sum_m"] for item in sampled)
            count = sum(by_horizon[key][item]["count"] for item in sampled)
            null_weighted_sum = 0.0
            for item in sampled:
                index = video_index[item]
                null_weighted_sum += (
                    null_result["_video_null_median"][key][index]
                    * by_horizon[key][item]["count"]
                )
            raw_boot[replicate] = raw_sum / max(count, 1)
            oracle_boot[replicate] = oracle_sum / max(count, 1)
            excess_boot[replicate] = excess_sum / max(count, 1)
            null_mean = null_weighted_sum / max(count, 1)
            coherence_boot[replicate] = 1.0 - excess_boot[replicate] / max(null_mean, 1e-12)
        output[key] = {
            "raw_2d_mae_m": _ci(raw_boot),
            "oracle_2d_mae_m": _ci(oracle_boot),
            "model_excess_2d_mae_m": _ci(excess_boot),
            "coherence_gain": _ci(coherence_boot),
        }
    return {
        "unit": "video",
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": 0.95,
        "coherence_null_summary": "sample-count-weighted mean of each selected video's permutation-median excess",
        "horizons": output,
    }


def _write_per_video(path: Path, rows: list[dict], null_result, video_ids):
    lookup = {str(video_id): index for index, video_id in enumerate(video_ids)}
    fields = list(rows[0]) + ["null_expected_excess_2d_mae_m", "coherence_gain"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            key = str(row["horizon_seconds"])
            index = lookup[row["video_id"]]
            null_mean = float(null_result["_video_null_median"][key][index])
            enriched = dict(row)
            enriched["null_expected_excess_2d_mae_m"] = null_mean
            enriched["coherence_gain"] = float(
                1.0 - row["excess_2d_mae_m"] / max(null_mean, 1e-12)
            )
            writer.writerow(enriched)


def evaluate_prediction_file(
    prediction_path: str | Path,
    gate_metrics_path: str | Path,
    output_dir: str | Path,
    permutations: int = 1000,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
    split_name: str | None = None,
) -> dict:
    prediction_path = Path(prediction_path).expanduser().resolve()
    gate_metrics_path = Path(gate_metrics_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = archive["prediction"].astype(np.float64)
        target = archive["target"].astype(np.float64)
        valid = archive["valid"].astype(bool)
        target_frames = archive["target_frame_indices"].astype(np.float64)
        video_ids = archive["video_id"].astype(str)
        target_names = tuple(archive["target_names"].astype(str).tolist())
        train_mean = archive["target_mean"].astype(np.float64)
        train_std = archive["target_std"].astype(np.float64)
        video_fps = float(archive["video_fps"].item())
        stored_split = str(archive["split_name"].item())
        model_name = str(archive["model_name"].item())
        probe_seed = int(archive["seed"].item())
    split_name = split_name or stored_split
    if split_name != stored_split:
        raise ValueError(f"Requested split {split_name!r}, archive contains {stored_split!r}")
    if target_names != TARGET_NAMES:
        raise ValueError(f"Target order mismatch: {target_names} != {TARGET_NAMES}")
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target, and valid must have identical [N,T,11] shapes")
    if prediction.ndim != 3 or prediction.shape[-1] != len(TARGET_NAMES):
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
    if len(set(video_ids.tolist())) != len(video_ids):
        raise ValueError("Prediction archive must contain one deterministic clip per video_id")
    with gate_metrics_path.open() as handle:
        gate = json.load(handle)
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
            raise RuntimeError(f"GT Gate protocol drift for {key}: {selected.get(key)!r} != {value!r}")
    thresholds = gate["train_frozen_thresholds"]
    point = _point_metrics(prediction, target, valid, train_mean, train_std)
    null, per_video = _accuracy_matched_null(
        prediction,
        target,
        valid,
        target_frames,
        video_ids,
        video_fps,
        thresholds,
        int(permutations),
        int(seed) + 41000,
    )
    trajectory, csv_rows = _aggregate_residuals(per_video)
    capture_dates = {
        str(video_id): _capture_date(gate["metadata"]["covla_root"], str(video_id))
        for video_id in video_ids
    }
    for row in csv_rows:
        row["capture_date"] = capture_dates[row["video_id"]]
    velocity_acceleration = _aggregate_va(per_video)
    labels, _ = _motion_strata(target, valid, target_frames, video_fps, thresholds)
    donors = _wrong_video_donors(labels, int(seed))
    hard_negative = {
        "definition": "Correct SE(2) rollout must rank ahead of five predeclared wrong-law alternatives; lower endpoint distance is better.",
        "model": _hard_negative_scores(
            prediction,
            valid,
            target,
            valid,
            target_frames,
            video_fps,
            thresholds,
            labels,
            donors,
        ),
        "gt_ceiling": _hard_negative_scores(
            target,
            valid,
            target,
            valid,
            target_frames,
            video_fps,
            thresholds,
            labels,
            donors,
        ),
    }
    bootstrap = _cluster_bootstrap(
        csv_rows, null, video_ids, int(bootstrap_replicates), int(seed) + 51000
    )
    per_video_path = output_dir / f"{split_name}_per_video.csv"
    _write_per_video(per_video_path, csv_rows, null, video_ids)
    null_arrays_path = output_dir / f"{split_name}_null_distribution.npz"
    np.savez_compressed(
        null_arrays_path,
        composite=null["_null_composite"],
        **{f"horizon_{key}": value for key, value in null["_null_arrays"].items()},
        **{f"va_horizon_{key}": value for key, value in null["_null_va_arrays"].items()},
    )
    for private_key in [key for key in null if key.startswith("_")]:
        del null[private_key]
    result = {
        "schema": 1,
        "experiment": "covla_independent_head_bank_physics_consistency",
        "model_name": model_name,
        "probe_seed": probe_seed,
        "split": split_name,
        "video_count": int(len(video_ids)),
        "capture_date_counts": {
            date: int(sum(value == date for value in capture_dates.values()))
            for date in sorted(set(capture_dates.values()))
        },
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "gt_gate_path": str(gate_metrics_path),
        "gt_gate_sha256": sha256_file(gate_metrics_path),
        "frozen_protocol": expected,
        "target_names": list(TARGET_NAMES),
        "point_metrics": point,
        "trajectory_se2": trajectory,
        "velocity_acceleration_auxiliary": velocity_acceleration,
        "accuracy_matched_null": null,
        "hard_negative_discrimination": hard_negative,
        "cluster_bootstrap": bootstrap,
        "outputs": {
            "per_video_csv": str(per_video_path),
            "null_distribution_npz": str(null_arrays_path),
        },
    }
    result_path = output_dir / f"{split_name}_consistency.json"
    atomic_json_dump(json_safe(result), result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split")
    args = parser.parse_args()
    result = evaluate_prediction_file(
        args.predictions,
        args.gt_gate,
        args.output_dir,
        permutations=args.permutations,
        bootstrap_replicates=args.bootstrap,
        seed=args.seed,
        split_name=args.split,
    )
    print(json.dumps(json_safe({
        "split": result["split"],
        "S_coh": result["accuracy_matched_null"]["composite"]["S_coh"],
        "output": str(Path(args.output_dir).resolve()),
    }), indent=2))


if __name__ == "__main__":
    main()
