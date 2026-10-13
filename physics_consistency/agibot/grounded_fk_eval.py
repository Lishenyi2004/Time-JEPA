#!/usr/bin/env python3
"""GT-grounded official-URDF evaluation for AgiBot prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .bank_common import (
        P_INDICES,
        Q_INDICES,
        TARGET_INDEX,
        TARGET_NAMES,
        atomic_json_dump,
        json_safe,
        sha256_file,
    )
    from .urdf_fk import (
        G2ForwardKinematics,
        apply_rigid_alignment,
        load_calibration,
        position_error_summary,
    )
except ImportError:
    from bank_common import (
        P_INDICES,
        Q_INDICES,
        TARGET_INDEX,
        TARGET_NAMES,
        atomic_json_dump,
        json_safe,
        sha256_file,
    )
    from urdf_fk import (
        G2ForwardKinematics,
        apply_rigid_alignment,
        load_calibration,
        position_error_summary,
    )


HARD_NEGATIVES = (
    "shuffled_joint_order",
    "wrong_arm_q",
    "wrong_arm_calibration",
    "one_token_time_shift",
    "four_token_time_shift",
    "matched_wrong_episode",
    "wrong_joint_sign",
    "wrong_zero_offset_0p25rad",
    "link7_no_end",
)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = None
    for array in arrays:
        candidate = np.all(np.isfinite(array), axis=-1) if array.ndim >= 3 else np.isfinite(array)
        mask = candidate if mask is None else mask & candidate
    return mask


def _scalar_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    train_mean: float,
    train_std: float,
) -> dict:
    mask = (np.asarray(valid) > 0) & np.isfinite(prediction) & np.isfinite(target)
    pred = np.asarray(prediction, dtype=np.float64)[mask]
    truth = np.asarray(target, dtype=np.float64)[mask]
    if not len(pred):
        return {"count": 0}
    delta = pred - truth
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta**2)))
    pred_center = pred - pred.mean()
    target_center = truth - truth.mean()
    covariance = float(np.sum(pred_center * target_center))
    pred_ss = float(np.sum(pred_center**2))
    target_ss = float(np.sum(target_center**2))
    pearson = covariance / np.sqrt(pred_ss * target_ss) if pred_ss > 0 and target_ss > 0 else np.nan
    pred_var = float(np.mean(pred_center**2))
    target_var = float(np.mean(target_center**2))
    ccc_denominator = pred_var + target_var + float((pred.mean() - truth.mean()) ** 2)
    ccc = 2.0 * float(np.mean(pred_center * target_center)) / ccc_denominator if ccc_denominator > 0 else np.nan
    r2 = 1.0 - float(np.sum(delta**2)) / target_ss if target_ss > 0 else np.nan
    baseline_sse = float(np.sum((truth - float(train_mean)) ** 2))
    skill = 1.0 - float(np.sum(delta**2)) / baseline_sse if baseline_sse > 0 else np.nan
    return {
        "count": int(len(pred)),
        "mae": mae,
        "rmse": rmse,
        "train_std_nrmse": rmse / max(float(train_std), 1e-12),
        "pearson": float(np.clip(pearson, -1.0, 1.0)),
        "ccc": float(np.clip(ccc, -1.0, 1.0)),
        "r2": float(r2),
        "skill_vs_train_mean": float(skill),
    }


def scalar_regression_metrics(payload: dict, fk: G2ForwardKinematics) -> dict:
    output = {}
    for index, name in enumerate(TARGET_NAMES):
        item = _scalar_metrics(
            payload["prediction"][..., index],
            payload["target"][..., index],
            payload["valid"][..., index],
            payload["target_mean"][index],
            payload["target_std"][index],
        )
        if name.startswith("q_") and item.get("count", 0):
            arm = "left" if name.startswith("q_left") else "right"
            joint = int(name.rsplit("_", 1)[1])
            lower, upper = fk.joint_limits(arm)[joint]
            values = payload["prediction"][..., index]
            mask = (payload["valid"][..., index] > 0) & np.isfinite(values)
            item["joint_limit_lower_rad"] = float(lower)
            item["joint_limit_upper_rad"] = float(upper)
            item["joint_limit_violation_rate"] = float(
                np.mean((values[mask] < lower) | (values[mask] > upper))
            )
        output[name] = item
    return output


def _aligned_fk(
    fk: G2ForwardKinematics,
    q: np.ndarray,
    arm: str,
    calibration: dict,
    include_end: bool = True,
) -> np.ndarray:
    return apply_rigid_alignment(
        fk.forward(np.asarray(q, dtype=np.float64), arm, include_end=include_end),
        calibration,
    )


def _distance(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(np.asarray(valid).shape, np.nan, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & _finite_mask(prediction, target)
    result[mask] = np.linalg.norm(prediction[mask] - target[mask], axis=-1)
    return result


def _position_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    workspace_span_m: float,
) -> dict:
    mask = np.asarray(valid, dtype=bool) & _finite_mask(prediction, target)
    summary = position_error_summary(prediction[mask], target[mask])
    if not summary.get("count", 0):
        return summary
    delta = prediction[mask] - target[mask]
    summary["coordinate_mae_m"] = np.mean(np.abs(delta), axis=0).tolist()
    summary["coordinate_rmse_m"] = np.sqrt(np.mean(delta**2, axis=0)).tolist()
    xz_distance = np.linalg.norm(delta[:, [0, 2]], axis=1)
    summary["xz_mean_euclidean_m"] = float(np.mean(xz_distance))
    summary["xz_p95_euclidean_m"] = float(np.quantile(xz_distance, 0.95))
    summary["workspace_normalized_mean"] = summary["mean_euclidean_m"] / max(
        float(workspace_span_m), 1e-12
    )
    return summary


def _active_mask(q_gt: np.ndarray, q_valid: np.ndarray, threshold: float) -> np.ndarray:
    active = np.zeros(q_valid.shape, dtype=bool)
    pair_valid = q_valid[:, 1:] & q_valid[:, :-1]
    delta = np.linalg.norm(q_gt[:, 1:] - q_gt[:, :-1], axis=-1)
    active[:, 1:] = pair_valid & (delta >= float(threshold))
    return active


def _row_activity(q_gt: np.ndarray, q_valid: np.ndarray) -> np.ndarray:
    output = np.zeros(len(q_gt), dtype=np.float64)
    for index in range(len(q_gt)):
        valid = q_valid[index, 1:] & q_valid[index, :-1]
        delta = np.linalg.norm(q_gt[index, 1:] - q_gt[index, :-1], axis=-1)
        output[index] = float(np.mean(delta[valid])) if np.any(valid) else 0.0
    return output


def _matched_wrong_episode_donors(payload: dict, activity: np.ndarray) -> np.ndarray:
    task_ids = np.asarray(payload["task_id"])
    episodes = np.asarray(payload["episode_key"]).astype(str)
    donors = np.full(len(task_ids), -1, dtype=np.int64)
    for task in np.unique(task_ids):
        members = np.flatnonzero(task_ids == task)
        for index in members:
            candidates = members[episodes[members] != episodes[index]]
            if len(candidates):
                distance = np.abs(activity[candidates] - activity[index])
                donors[index] = int(candidates[np.lexsort((candidates, distance))[0]])
    return donors


def _shift_tokens(q: np.ndarray, q_valid: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    shifted = np.full_like(q, np.nan, dtype=np.float64)
    valid = np.zeros_like(q_valid, dtype=bool)
    if offset > 0 and offset < q.shape[1]:
        shifted[:, :-offset] = q[:, offset:]
        valid[:, :-offset] = q_valid[:, offset:]
    elif offset < 0 and -offset < q.shape[1]:
        shifted[:, -offset:] = q[:, :offset]
        valid[:, -offset:] = q_valid[:, :offset]
    return shifted, valid


def _candidate_scores(
    payload: dict,
    fk: G2ForwardKinematics,
    calibrations: dict,
    arm: str,
    q_source: np.ndarray,
    source_valid: np.ndarray,
    donors: np.ndarray,
    source_kind: str,
) -> dict[str, np.ndarray]:
    p_idx = P_INDICES[arm]
    p_gt = payload["target"][..., p_idx]
    p_valid = np.all(payload["valid"][..., p_idx] > 0, axis=-1)
    other = "right" if arm == "left" else "left"
    other_idx = Q_INDICES[other]
    if source_kind not in {"prediction", "target"}:
        raise ValueError(f"Unknown candidate source kind: {source_kind}")
    source_is_prediction = source_kind == "prediction"
    other_q = (
        payload["prediction"][..., other_idx]
        if source_is_prediction
        else payload["target"][..., other_idx]
    )
    other_valid = np.all(payload["valid"][..., other_idx] > 0, axis=-1)

    variants: dict[str, tuple[np.ndarray, np.ndarray, str, dict, bool]] = {
        "correct": (q_source, source_valid, arm, calibrations[arm], True),
        "shuffled_joint_order": (q_source[..., ::-1], source_valid, arm, calibrations[arm], True),
        "wrong_arm_q": (other_q, other_valid, arm, calibrations[arm], True),
        "wrong_arm_calibration": (q_source, source_valid, arm, calibrations[other], True),
        "wrong_joint_sign": (-q_source, source_valid, arm, calibrations[arm], True),
        "wrong_zero_offset_0p25rad": (q_source + 0.25, source_valid, arm, calibrations[arm], True),
        "link7_no_end": (q_source, source_valid, arm, calibrations[arm], False),
    }
    for name, offset in (("one_token_time_shift", 1), ("four_token_time_shift", 4)):
        shifted, shifted_valid = _shift_tokens(q_source, source_valid, offset)
        variants[name] = (shifted, shifted_valid, arm, calibrations[arm], True)
    donor_q = np.full_like(q_source, np.nan)
    donor_valid = np.zeros_like(source_valid)
    available = donors >= 0
    donor_q[available] = q_source[donors[available]]
    donor_valid[available] = source_valid[donors[available]]
    variants["matched_wrong_episode"] = (
        donor_q,
        donor_valid,
        arm,
        calibrations[arm],
        True,
    )

    scores = {}
    for name, (q_value, q_mask, chain_arm, calibration, include_end) in variants.items():
        positions = _aligned_fk(fk, q_value, chain_arm, calibration, include_end=include_end)
        scores[name] = _distance(positions, p_gt, p_valid & q_mask)
    return scores


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _binary_discrimination(correct: np.ndarray, negative: np.ndarray) -> dict:
    valid = np.isfinite(correct) & np.isfinite(negative)
    correct, negative = correct[valid], negative[valid]
    if not len(correct):
        return {"count": 0}
    labels = np.concatenate([np.ones(len(correct)), np.zeros(len(negative))])
    # Higher score means more likely to be the correct law.
    scores = -np.concatenate([correct, negative])
    ranks = _average_ranks(scores)
    positives = labels == 1
    n_pos, n_neg = int(np.sum(positives)), int(np.sum(~positives))
    auroc = (
        float(np.sum(ranks[positives])) - n_pos * (n_pos + 1) / 2.0
    ) / max(n_pos * n_neg, 1)
    margin = negative - correct
    return {
        "count": int(len(correct)),
        "auroc": float(auroc),
        "paired_strict_accuracy": float(np.mean(margin > 0)),
        "paired_tie_aware_accuracy": float(np.mean((margin > 0) + 0.5 * (margin == 0))),
        "mean_margin_negative_minus_correct_m": float(np.mean(margin)),
        "median_margin_negative_minus_correct_m": float(np.median(margin)),
    }


def _hard_negative_summary(
    scores: dict[str, np.ndarray],
    active: np.ndarray,
    row_mask: np.ndarray | None = None,
) -> dict:
    if row_mask is None:
        row_mask = np.ones(scores["correct"].shape[0], dtype=bool)
    correct = scores["correct"][row_mask]
    active_flat = active[row_mask].reshape(-1)
    correct_flat = correct.reshape(-1)
    pairwise = {}
    active_aurocs = []
    overall_aurocs = []
    candidates = []
    for name in HARD_NEGATIVES:
        negative = scores[name][row_mask].reshape(-1)
        overall = _binary_discrimination(correct_flat, negative)
        conditioned = _binary_discrimination(correct_flat[active_flat], negative[active_flat])
        pairwise[name] = {"overall": overall, "active_motion": conditioned}
        if overall.get("count", 0):
            overall_aurocs.append(overall["auroc"])
        if conditioned.get("count", 0):
            active_aurocs.append(conditioned["auroc"])
        candidates.append(negative.reshape(correct.shape))
    matrix = np.stack(candidates, axis=-1)
    common = np.isfinite(correct) & np.all(np.isfinite(matrix), axis=-1)
    if np.any(common):
        correct_values = correct[common]
        negatives = matrix[common]
        strict = np.mean(np.all(correct_values[:, None] < negatives, axis=1))
        all_values = np.concatenate([correct_values[:, None], negatives], axis=1)
        minima = np.min(all_values, axis=1, keepdims=True)
        tied = np.isclose(all_values, minima, rtol=1e-12, atol=1e-12)
        tie_recall = np.mean(tied[:, 0] / np.sum(tied, axis=1))
        rank = 1.0 + np.sum(negatives < correct_values[:, None], axis=1)
        rank += 0.5 * np.sum(negatives == correct_values[:, None], axis=1)
        mrr = np.mean(1.0 / rank)
        hardest_margin = np.mean(np.min(negatives, axis=1) - correct_values)
    else:
        strict = tie_recall = mrr = hardest_margin = np.nan
    return {
        "retrieval_count": int(np.sum(common)),
        "strict_recall_at_1": float(strict),
        "tie_aware_expected_recall_at_1": float(tie_recall),
        "midrank_mrr": float(mrr),
        "overall_macro_auroc": float(np.mean(overall_aurocs)) if overall_aurocs else np.nan,
        "conditioned_macro_auroc": float(np.mean(active_aurocs)) if active_aurocs else np.nan,
        "mean_margin_hardest_negative_minus_correct_m": float(hardest_margin),
        "pairwise": pairwise,
    }


def _arm_state(
    payload: dict,
    fk: G2ForwardKinematics,
    calibrations: dict,
    gate: dict,
    arm: str,
) -> dict:
    q_idx, p_idx = Q_INDICES[arm], P_INDICES[arm]
    q_pred = payload["prediction"][..., q_idx]
    q_gt = payload["target"][..., q_idx]
    p_pred = payload["prediction"][..., p_idx]
    p_gt = payload["target"][..., p_idx]
    q_valid = np.all(payload["valid"][..., q_idx] > 0, axis=-1)
    p_valid = np.all(payload["valid"][..., p_idx] > 0, axis=-1)
    valid = q_valid & p_valid
    calibration = calibrations[arm]
    fk_pred = _aligned_fk(fk, q_pred, arm, calibration)
    fk_gt = _aligned_fk(fk, q_gt, arm, calibration)
    q_mean = np.broadcast_to(payload["target_mean"][q_idx], q_pred.shape)
    q_zero = np.zeros_like(q_pred)
    fk_mean = _aligned_fk(fk, q_mean, arm, calibration)
    fk_zero = _aligned_fk(fk, q_zero, arm, calibration)
    threshold = float(gate["arms"][arm]["active_thresholds"]["token_delta_l2_rad_q75"])
    active = _active_mask(q_gt, q_valid, threshold) & p_valid
    activity = _row_activity(q_gt, q_valid)
    donors = _matched_wrong_episode_donors(payload, activity)
    pred_scores = _candidate_scores(
        payload, fk, calibrations, arm, q_pred, q_valid, donors, "prediction"
    )
    gt_scores = _candidate_scores(
        payload, fk, calibrations, arm, q_gt, q_valid, donors, "target"
    )
    return {
        "q_pred": q_pred,
        "q_gt": q_gt,
        "p_pred": p_pred,
        "p_gt": p_gt,
        "q_valid": q_valid,
        "p_valid": p_valid,
        "valid": valid,
        "active": active,
        "fk_pred": fk_pred,
        "fk_gt": fk_gt,
        "fk_mean": fk_mean,
        "fk_zero": fk_zero,
        "pred_scores": pred_scores,
        "gt_scores": gt_scores,
        "workspace_span_m": float(gate["arms"][arm]["workspace"]["robust_diagonal_m"]),
        "active_threshold": threshold,
    }


def _aggregate_q_metrics(payload: dict, arm: str, row_mask: np.ndarray) -> dict:
    items = []
    for index in Q_INDICES[arm]:
        item = _scalar_metrics(
            payload["prediction"][row_mask, :, index],
            payload["target"][row_mask, :, index],
            payload["valid"][row_mask, :, index],
            payload["target_mean"][index],
            payload["target_std"][index],
        )
        items.append(item)
    usable = [item for item in items if item.get("count", 0)]
    return {
        "macro_train_std_nrmse": float(np.mean([item["train_std_nrmse"] for item in usable])) if usable else np.nan,
        "macro_skill_vs_train_mean": float(np.mean([item["skill_vs_train_mean"] for item in usable])) if usable else np.nan,
        "joint_count": len(usable),
    }


def _summarize_arm(payload: dict, state: dict, arm: str, row_mask: np.ndarray) -> dict:
    valid = state["valid"][row_mask]
    active = state["active"][row_mask]
    span = state["workspace_span_m"]
    grounded = _position_metrics(state["fk_pred"][row_mask], state["p_gt"][row_mask], valid, span)
    direct = _position_metrics(state["p_pred"][row_mask], state["p_gt"][row_mask], state["p_valid"][row_mask], span)
    oracle = _position_metrics(state["fk_gt"][row_mask], state["p_gt"][row_mask], valid, span)
    train_mean = _position_metrics(state["fk_mean"][row_mask], state["p_gt"][row_mask], valid, span)
    zero = _position_metrics(state["fk_zero"][row_mask], state["p_gt"][row_mask], valid, span)
    closure = _position_metrics(state["p_pred"][row_mask], state["fk_pred"][row_mask], valid, span)
    active_grounded = _position_metrics(
        state["fk_pred"][row_mask], state["p_gt"][row_mask], active, span
    )
    model_sse = np.nansum(state["pred_scores"]["correct"][row_mask] ** 2)
    baseline_sse = np.nansum(
        _distance(state["fk_mean"][row_mask], state["p_gt"][row_mask], valid) ** 2
    )
    if grounded.get("count", 0) and oracle.get("count", 0):
        grounded["mean_oracle_gap_m"] = grounded["mean_euclidean_m"] - oracle["mean_euclidean_m"]
    return {
        "q_point_macro": _aggregate_q_metrics(payload, arm, row_mask),
        "grounded_fk_to_gt": grounded,
        "grounded_fk_to_gt_active_motion": active_grounded,
        "direct_endpoint_to_gt": direct,
        "gt_oracle_floor": oracle,
        "prediction_only_closure_diagnostic": closure,
        "controls": {
            "train_mean_q_to_gt": train_mean,
            "all_zero_q_to_gt": zero,
            "fk_skill_vs_train_mean": float(1.0 - model_sse / baseline_sse) if baseline_sse > 0 else np.nan,
        },
        "hard_negatives": _hard_negative_summary(state["pred_scores"], state["active"], row_mask),
        "gt_q_hard_negative_ceiling": _hard_negative_summary(state["gt_scores"], state["active"], row_mask),
        "active_threshold_token_delta_l2_rad": state["active_threshold"],
    }


def _flatten_row(prefix: dict, summary: dict) -> dict:
    def value(path, default=np.nan):
        current = summary
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    return {
        **prefix,
        "q_macro_nrmse": value("q_point_macro.macro_train_std_nrmse"),
        "q_macro_skill": value("q_point_macro.macro_skill_vs_train_mean"),
        "fk_mean_m": value("grounded_fk_to_gt.mean_euclidean_m"),
        "fk_xz_mean_m": value("grounded_fk_to_gt.xz_mean_euclidean_m"),
        "fk_p95_m": value("grounded_fk_to_gt.p95_euclidean_m"),
        "fk_active_mean_m": value("grounded_fk_to_gt_active_motion.mean_euclidean_m"),
        "direct_mean_m": value("direct_endpoint_to_gt.mean_euclidean_m"),
        "direct_xz_mean_m": value("direct_endpoint_to_gt.xz_mean_euclidean_m"),
        "oracle_mean_m": value("gt_oracle_floor.mean_euclidean_m"),
        "closure_mean_m": value("prediction_only_closure_diagnostic.mean_euclidean_m"),
        "train_mean_control_m": value("controls.train_mean_q_to_gt.mean_euclidean_m"),
        "zero_control_m": value("controls.all_zero_q_to_gt.mean_euclidean_m"),
        "fk_skill_vs_train_mean": value("controls.fk_skill_vs_train_mean"),
        "law_conditioned_macro_auroc": value("hard_negatives.conditioned_macro_auroc"),
        "law_overall_macro_auroc": value("hard_negatives.overall_macro_auroc"),
        "law_strict_recall_at_1": value("hard_negatives.strict_recall_at_1"),
        "valid_tokens": value("grounded_fk_to_gt.count", 0),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def load_prediction_file(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    names = tuple(str(item) for item in payload["target_names"].tolist())
    if names != TARGET_NAMES:
        raise ValueError(f"Prediction target schema mismatch: {names} != {TARGET_NAMES}")
    for key in ("prediction", "target", "valid"):
        if payload[key].ndim != 3 or payload[key].shape[-1] != len(TARGET_NAMES):
            raise ValueError(f"{key} must be [N,T,{len(TARGET_NAMES)}], got {payload[key].shape}")
    if not (payload["prediction"].shape == payload["target"].shape == payload["valid"].shape):
        raise ValueError("prediction/target/valid shapes differ")
    rows = payload["prediction"].shape[0]
    for key in ("episode_key", "window_key", "task_id", "view_key"):
        if key not in payload or len(payload[key]) != rows:
            raise ValueError(f"Prediction artifact needs row metadata {key!r}")
    payload["path"] = str(path)
    return payload


def evaluate_prediction_file(
    prediction_path: str | Path,
    gt_gate_metrics: str | Path,
    output_dir: str | Path,
    split_name: str | None = None,
) -> dict:
    payload = load_prediction_file(prediction_path)
    gate_path = Path(gt_gate_metrics).expanduser().resolve()
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "PASS":
        raise RuntimeError(f"Refusing physical evaluation because GT gate did not pass: {gate_path}")
    urdf_path = Path(gate["urdf"]["path"])
    if sha256_file(urdf_path) != gate["urdf"]["sha256"]:
        raise RuntimeError("URDF changed after the GT gate")
    calibration_path = Path(gate["calibration_path"])
    calibrations = load_calibration(calibration_path)
    fk = G2ForwardKinematics(urdf_path)
    if split_name is None:
        split_name = str(payload.get("split_name", np.asarray("unknown")).item())
    row_count = payload["prediction"].shape[0]
    states = {
        arm: _arm_state(payload, fk, calibrations, gate, arm)
        for arm in ("left", "right")
    }
    all_rows = np.ones(row_count, dtype=bool)
    overall = {
        arm: _summarize_arm(payload, states[arm], arm, all_rows)
        for arm in ("left", "right")
    }
    episode_rows, task_rows = [], []
    for episode in sorted(set(payload["episode_key"].astype(str).tolist())):
        mask = payload["episode_key"].astype(str) == episode
        for arm in ("left", "right"):
            summary = _summarize_arm(payload, states[arm], arm, mask)
            episode_rows.append(
                _flatten_row(
                    {
                        "split": split_name,
                        "view": str(payload["view_key"][np.flatnonzero(mask)[0]]),
                        "episode_key": episode,
                        "task_id": int(payload["task_id"][np.flatnonzero(mask)[0]]),
                        "arm": arm,
                    },
                    summary,
                )
            )
    for task in sorted(set(payload["task_id"].astype(int).tolist())):
        mask = payload["task_id"].astype(int) == task
        for arm in ("left", "right"):
            summary = _summarize_arm(payload, states[arm], arm, mask)
            task_rows.append(
                _flatten_row(
                    {"split": split_name, "task_id": int(task), "arm": arm}, summary
                )
            )
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / f"{split_name}_per_episode.csv"
    task_path = output_dir / f"{split_name}_per_task.csv"
    _write_csv(episode_path, episode_rows)
    _write_csv(task_path, task_rows)
    result = {
        "schema": 1,
        "protocol": "agibot_official_g2_urdf_gt_grounded_v1",
        "split": split_name,
        "view_keys": sorted(set(payload["view_key"].astype(str).tolist())),
        "row_count": row_count,
        "token_count": int(np.prod(payload["prediction"].shape[:2])),
        "prediction_artifact": {
            "path": payload["path"],
            "sha256": sha256_file(payload["path"]),
        },
        "gt_gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
        "urdf": gate["urdf"],
        "scalar_regression": scalar_regression_metrics(payload, fk),
        "arms": overall,
        "per_episode_csv": str(episode_path),
        "per_task_csv": str(task_path),
        "interpretation_guard": (
            "grounded_fk_to_gt is primary; prediction_only_closure_diagnostic is never a main score"
        ),
    }
    metrics_path = output_dir / f"{split_name}_grounded_fk.json"
    atomic_json_dump(json_safe(result), metrics_path)
    result["metrics_path"] = str(metrics_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name")
    args = parser.parse_args()
    result = evaluate_prediction_file(
        args.predictions, args.gt_gate, args.output_dir, split_name=args.split_name
    )
    print(json.dumps(json_safe({"metrics": result["metrics_path"], "arms": result["arms"]}), indent=2))


if __name__ == "__main__":
    main()
