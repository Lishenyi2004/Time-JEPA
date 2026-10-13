"""Encoder checkpoint loading, masked loss, evaluation, and checkpoint saving."""

from pathlib import Path

import torch

from probe_utils import reduce_sum, torch_load, tqdm, unwrap_model


def _strip_encoder_key(key):
    for prefix in ("module.backbone.", "module.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def _checkpoint_candidates(checkpoint, requested_key):
    if requested_key != "auto":
        return [requested_key]
    return ["target_encoder", "encoder", "ema_encoder"]


def load_encoder(model, checkpoint_path, checkpoint_key="auto"):
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    state = None
    selected_key = None
    for key in _checkpoint_candidates(checkpoint, checkpoint_key):
        if key in checkpoint:
            state = checkpoint[key]
            selected_key = key
            break
    if state is None:
        available = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
        raise KeyError(
            f"Cannot find checkpoint key {checkpoint_key!r} in {checkpoint_path}; "
            f"available top-level keys: {available[:20]}"
        )
    state = {_strip_encoder_key(k): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    return {
        "checkpoint_key": selected_key,
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }


def masked_mse(pred, target, valid, balance_time_steps=False):
    if pred.shape != target.shape:
        raise RuntimeError(f"Prediction and target shapes must match, got {pred.shape=} {target.shape=}")
    loss = (pred - target) ** 2
    loss = loss * valid
    if balance_time_steps:
        if loss.ndim != 3:
            raise RuntimeError(
                "Time-balanced masked MSE expects [B, H, D] tensors, got "
                f"{tuple(loss.shape)}"
            )
        per_step_count = valid.sum(dim=(0, 2))
        per_step_loss = loss.sum(dim=(0, 2)) / per_step_count.clamp_min(1.0)
        valid_steps = per_step_count > 0
        return (
            (per_step_loss * valid_steps).sum()
            / valid_steps.sum().clamp_min(1)
        )
    denom = valid.sum().clamp_min(1.0)
    return loss.sum() / denom


def _regression_metrics_from_sums(
    abs_error_sum,
    sq_error_sum,
    pred_sum,
    target_sum,
    pred_sq_sum,
    target_sq_sum,
    pred_target_sum,
    valid_sum,
    train_target_mean,
    train_target_std,
):
    """Compute per-target regression and agreement metrics from sufficient statistics.

    ``r2`` uses the mean of the current evaluation split as its constant
    baseline. ``skill_train_mean`` instead uses the probe-training target mean,
    which is the constant baseline available at deployment time.
    """
    count = valid_sum.clamp_min(1.0)
    mse = sq_error_sum / count
    mae = abs_error_sum / count
    rmse = torch.sqrt(mse.clamp_min(0.0))
    nrmse = rmse / train_target_std.clamp_min(1e-6)

    pred_mean = pred_sum / count
    target_mean = target_sum / count
    covariance_sum = pred_target_sum - (pred_sum * target_sum) / count
    pred_var_sum = (pred_sq_sum - (pred_sum * pred_sum) / count).clamp_min(0.0)
    target_var_sum = (target_sq_sum - (target_sum * target_sum) / count).clamp_min(0.0)

    pearson_denom = torch.sqrt((pred_var_sum * target_var_sum).clamp_min(1e-12))
    pearson = covariance_sum / pearson_denom
    pearson = torch.where(valid_sum > 1.0, pearson.clamp(-1.0, 1.0), torch.zeros_like(pearson))

    ccc_denom = pred_var_sum + target_var_sum + count * (pred_mean - target_mean).square()
    ccc = 2.0 * covariance_sum / ccc_denom.clamp_min(1e-12)
    ccc = torch.where(
        (valid_sum > 1.0) & (ccc_denom > 1e-12),
        ccc.clamp(-1.0, 1.0),
        torch.zeros_like(ccc),
    )

    r2 = 1.0 - sq_error_sum / target_var_sum.clamp_min(1e-12)
    r2 = torch.where(
        (valid_sum > 1.0) & (target_var_sum > 1e-12),
        r2,
        torch.zeros_like(r2),
    )

    train_target_mean = train_target_mean.to(device=target_sum.device, dtype=target_sum.dtype)
    train_mean_baseline_sse = (
        target_sq_sum
        - 2.0 * train_target_mean * target_sum
        + train_target_mean.square() * valid_sum
    ).clamp_min(0.0)
    skill_train_mean = 1.0 - sq_error_sum / train_mean_baseline_sse.clamp_min(1e-12)
    skill_train_mean = torch.where(
        (valid_sum > 0.0) & (train_mean_baseline_sse > 1e-12),
        skill_train_mean,
        torch.zeros_like(skill_train_mean),
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "pearson": pearson,
        "ccc": ccc,
        "r2": r2,
        "skill_train_mean": skill_train_mean,
    }


def _baseline_error_metrics(abs_error_sum, sq_error_sum, valid_sum, train_target_std):
    count = valid_sum.clamp_min(1.0)
    rmse = torch.sqrt((sq_error_sum / count).clamp_min(0.0))
    return {
        "mae": abs_error_sum / count,
        "rmse": rmse,
        "nrmse": rmse / train_target_std.clamp_min(1e-6),
        "valid_count": valid_sum,
    }


def _add_paired_baseline_metrics(
    baseline_metrics,
    baseline_abs_error_sum,
    baseline_sq_error_sum,
    model_abs_error_sum,
    model_sq_error_sum,
    valid_sum,
    target_std,
):
    """Attach model errors and skill for a baseline on its valid subset.

    Keeping this in one place makes privileged (ground-truth current state) and
    estimated (video-derived current state) persistence use exactly the same
    paired-mask and metric definitions.
    """
    paired_model_metrics = _baseline_error_metrics(
        model_abs_error_sum,
        model_sq_error_sum,
        valid_sum,
        target_std,
    )
    for name in ("mae", "rmse", "nrmse"):
        baseline_metrics[f"paired_model_{name}"] = paired_model_metrics[name]
    baseline_metrics["model_skill"] = torch.where(
        baseline_sq_error_sum > 1e-12,
        1.0 - model_sq_error_sum / baseline_sq_error_sum.clamp_min(1e-12),
        torch.zeros_like(baseline_sq_error_sum),
    )
    return baseline_metrics


def _serialize_tensor_metrics(metrics):
    return {
        name: value.detach().cpu().tolist()
        for name, value in metrics.items()
    }


def _select_horizon_metrics(metrics, horizon_index):
    return {
        name: value[horizon_index].detach().cpu().tolist()
        for name, value in metrics.items()
    }


def _macro_horizon_metrics(metrics):
    return {
        name: value.mean(dim=0).detach().cpu().tolist()
        for name, value in metrics.items()
    }


def evaluate(
    encoder,
    probe,
    loader,
    device,
    target_mean,
    target_std,
    distributed=False,
    is_main=True,
    desc="eval",
    future_horizons_seconds=None,
    estimated_persistence_probe=None,
    estimated_persistence_target_mean=None,
    estimated_persistence_target_std=None,
):
    encoder.eval()
    probe.eval()
    multi_horizon = target_mean.ndim == 2
    if target_mean.ndim not in (1, 2):
        raise ValueError(
            "target_mean must have shape [D] or [H, D], got "
            f"{tuple(target_mean.shape)}"
        )
    if target_std.shape != target_mean.shape:
        raise ValueError(
            "target_mean and target_std must have matching shapes, got "
            f"{tuple(target_mean.shape)} and {tuple(target_std.shape)}"
        )
    if multi_horizon:
        if future_horizons_seconds is None:
            raise ValueError(
                "future_horizons_seconds is required when target stats have shape [H, D]"
            )
        future_horizons_seconds = [float(value) for value in future_horizons_seconds]
        if len(future_horizons_seconds) != target_mean.size(0):
            raise ValueError(
                "Number of future horizons must match target stats: "
                f"{len(future_horizons_seconds)} vs {target_mean.size(0)}"
            )
    sum_dims = (0,) if multi_horizon else (0, 1)
    total_loss_sum = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    abs_error_sum = torch.zeros_like(target_mean, device=device)
    sq_error_sum = torch.zeros_like(target_mean, device=device)
    pred_sum = torch.zeros_like(target_mean, device=device)
    target_sum = torch.zeros_like(target_mean, device=device)
    pred_sq_sum = torch.zeros_like(target_mean, device=device)
    target_sq_sum = torch.zeros_like(target_mean, device=device)
    pred_target_sum = torch.zeros_like(target_mean, device=device)
    valid_sum = torch.zeros_like(target_mean, device=device)
    train_mean_abs_error_sum = torch.zeros_like(target_mean, device=device)
    train_mean_sq_error_sum = torch.zeros_like(target_mean, device=device)
    persistence_abs_error_sum = torch.zeros_like(target_mean, device=device)
    persistence_sq_error_sum = torch.zeros_like(target_mean, device=device)
    model_abs_error_on_persistence_sum = torch.zeros_like(target_mean, device=device)
    model_sq_error_on_persistence_sum = torch.zeros_like(target_mean, device=device)
    persistence_valid_sum = torch.zeros_like(target_mean, device=device)
    persistence_batch_count = torch.zeros((), device=device)
    estimated_persistence_abs_error_sum = torch.zeros_like(target_mean, device=device)
    estimated_persistence_sq_error_sum = torch.zeros_like(target_mean, device=device)
    model_abs_error_on_estimated_persistence_sum = torch.zeros_like(target_mean, device=device)
    model_sq_error_on_estimated_persistence_sum = torch.zeros_like(target_mean, device=device)
    estimated_persistence_valid_sum = torch.zeros_like(target_mean, device=device)
    estimated_persistence_batch_count = torch.zeros((), device=device)

    if estimated_persistence_probe is not None:
        if not multi_horizon:
            raise ValueError("estimated persistence requires a future_state evaluation")
        if estimated_persistence_target_mean is None or estimated_persistence_target_std is None:
            raise ValueError(
                "estimated_persistence_target_mean/std are required with "
                "estimated_persistence_probe"
            )
        estimated_persistence_probe.eval()
        estimated_persistence_target_mean = torch.as_tensor(
            estimated_persistence_target_mean, device=device, dtype=target_mean.dtype
        )
        estimated_persistence_target_std = torch.as_tensor(
            estimated_persistence_target_std, device=device, dtype=target_mean.dtype
        )
        if estimated_persistence_target_mean.ndim != 1 or estimated_persistence_target_std.shape != estimated_persistence_target_mean.shape:
            raise ValueError("estimated persistence stats must have shape [D]")
        if estimated_persistence_target_mean.numel() != target_mean.size(-1):
            raise ValueError(
                "estimated persistence probe output dimension must match future targets: "
                f"{estimated_persistence_target_mean.numel()} vs {target_mean.size(-1)}"
            )

    with torch.inference_mode():
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not is_main)
        for batch in pbar:
            videos = batch["video"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            features = encoder(videos)
            preds = probe(features)
            loss = masked_mse(
                preds,
                targets,
                valid,
                balance_time_steps=multi_horizon,
            )
            total_loss_sum += loss.detach() * len(videos)
            total_count += len(videos)

            preds_raw = preds * target_std.to(device) + target_mean.to(device)
            targets_raw = targets * target_std.to(device) + target_mean.to(device)
            error = preds_raw - targets_raw
            abs_error_sum += (error.abs() * valid).sum(dim=sum_dims)
            sq_error_sum += ((error**2) * valid).sum(dim=sum_dims)
            pred_sum += (preds_raw * valid).sum(dim=sum_dims)
            target_sum += (targets_raw * valid).sum(dim=sum_dims)
            pred_sq_sum += ((preds_raw**2) * valid).sum(dim=sum_dims)
            target_sq_sum += ((targets_raw**2) * valid).sum(dim=sum_dims)
            pred_target_sum += ((preds_raw * targets_raw) * valid).sum(dim=sum_dims)
            valid_sum += valid.sum(dim=sum_dims)
            train_mean_for_batch = target_mean.to(device)
            if multi_horizon:
                train_mean_for_batch = train_mean_for_batch.unsqueeze(0)
            else:
                train_mean_for_batch = train_mean_for_batch.reshape(1, 1, -1)
            train_mean_error = train_mean_for_batch - targets_raw
            train_mean_abs_error_sum += (train_mean_error.abs() * valid).sum(dim=sum_dims)
            train_mean_sq_error_sum += ((train_mean_error**2) * valid).sum(dim=sum_dims)
            if "current_target" in batch:
                persistence_batch_count += 1
                current_target = batch["current_target"].to(device, non_blocking=True)
                current_valid = batch["current_valid"].to(device, non_blocking=True)
                persistence_valid = valid * current_valid
                persistence_error = current_target - targets_raw
                persistence_abs_error_sum += (
                    persistence_error.abs() * persistence_valid
                ).sum(dim=sum_dims)
                persistence_sq_error_sum += (
                    (persistence_error**2) * persistence_valid
                ).sum(dim=sum_dims)
                model_abs_error_on_persistence_sum += (
                    error.abs() * persistence_valid
                ).sum(dim=sum_dims)
                model_sq_error_on_persistence_sum += (
                    (error**2) * persistence_valid
                ).sum(dim=sum_dims)
                persistence_valid_sum += persistence_valid.sum(dim=sum_dims)
            if estimated_persistence_probe is not None:
                current_pred = estimated_persistence_probe(features)
                if current_pred.ndim == 3:
                    current_pred = current_pred[:, -1, :]
                if current_pred.ndim != 2:
                    raise RuntimeError(
                        "estimated persistence probe must produce [B, T, D] or [B, D], "
                        f"got {tuple(current_pred.shape)}"
                    )
                current_pred_raw = (
                    current_pred * estimated_persistence_target_std.unsqueeze(0)
                    + estimated_persistence_target_mean.unsqueeze(0)
                )
                current_pred_raw = current_pred_raw.unsqueeze(1).expand(
                    -1, targets_raw.size(1), -1
                )
                # The estimated baseline gets its current state from video, so
                # it remains evaluable even when a ground-truth current label
                # is missing. Pair it with the future model on future-valid
                # entries only; privileged persistence has its own stricter
                # future_valid & current_valid mask above.
                estimated_valid = valid
                estimated_error = current_pred_raw - targets_raw
                estimated_persistence_abs_error_sum += (
                    estimated_error.abs() * estimated_valid
                ).sum(dim=sum_dims)
                estimated_persistence_sq_error_sum += (
                    (estimated_error**2) * estimated_valid
                ).sum(dim=sum_dims)
                model_abs_error_on_estimated_persistence_sum += (
                    error.abs() * estimated_valid
                ).sum(dim=sum_dims)
                model_sq_error_on_estimated_persistence_sum += (
                    (error**2) * estimated_valid
                ).sum(dim=sum_dims)
                estimated_persistence_valid_sum += estimated_valid.sum(dim=sum_dims)
                estimated_persistence_batch_count += 1
            pbar.set_postfix(loss=f"{(total_loss_sum / total_count.clamp_min(1.0)).item():.4f}")

    total_loss_sum = reduce_sum(total_loss_sum, distributed)
    total_count = reduce_sum(total_count, distributed)
    abs_error_sum = reduce_sum(abs_error_sum, distributed)
    sq_error_sum = reduce_sum(sq_error_sum, distributed)
    pred_sum = reduce_sum(pred_sum, distributed)
    target_sum = reduce_sum(target_sum, distributed)
    pred_sq_sum = reduce_sum(pred_sq_sum, distributed)
    target_sq_sum = reduce_sum(target_sq_sum, distributed)
    pred_target_sum = reduce_sum(pred_target_sum, distributed)
    valid_sum = reduce_sum(valid_sum, distributed)
    train_mean_abs_error_sum = reduce_sum(train_mean_abs_error_sum, distributed)
    train_mean_sq_error_sum = reduce_sum(train_mean_sq_error_sum, distributed)
    persistence_abs_error_sum = reduce_sum(persistence_abs_error_sum, distributed)
    persistence_sq_error_sum = reduce_sum(persistence_sq_error_sum, distributed)
    model_abs_error_on_persistence_sum = reduce_sum(
        model_abs_error_on_persistence_sum,
        distributed,
    )
    model_sq_error_on_persistence_sum = reduce_sum(model_sq_error_on_persistence_sum, distributed)
    persistence_valid_sum = reduce_sum(persistence_valid_sum, distributed)
    persistence_batch_count = reduce_sum(persistence_batch_count, distributed)
    estimated_persistence_abs_error_sum = reduce_sum(
        estimated_persistence_abs_error_sum, distributed
    )
    estimated_persistence_sq_error_sum = reduce_sum(
        estimated_persistence_sq_error_sum, distributed
    )
    model_abs_error_on_estimated_persistence_sum = reduce_sum(
        model_abs_error_on_estimated_persistence_sum, distributed
    )
    model_sq_error_on_estimated_persistence_sum = reduce_sum(
        model_sq_error_on_estimated_persistence_sum, distributed
    )
    estimated_persistence_valid_sum = reduce_sum(
        estimated_persistence_valid_sum, distributed
    )
    estimated_persistence_batch_count = reduce_sum(
        estimated_persistence_batch_count, distributed
    )

    metrics = _regression_metrics_from_sums(
        abs_error_sum=abs_error_sum,
        sq_error_sum=sq_error_sum,
        pred_sum=pred_sum,
        target_sum=target_sum,
        pred_sq_sum=pred_sq_sum,
        target_sq_sum=target_sq_sum,
        pred_target_sum=pred_target_sum,
        valid_sum=valid_sum,
        train_target_mean=target_mean,
        train_target_std=target_std.to(device),
    )
    train_mean_metrics = _baseline_error_metrics(
        train_mean_abs_error_sum,
        train_mean_sq_error_sum,
        valid_sum,
        target_std.to(device),
    )
    persistence_metrics = None
    if persistence_batch_count.item() > 0:
        persistence_metrics = _baseline_error_metrics(
            persistence_abs_error_sum,
            persistence_sq_error_sum,
            persistence_valid_sum,
            target_std.to(device),
        )
        persistence_metrics = _add_paired_baseline_metrics(
            persistence_metrics,
            persistence_abs_error_sum,
            persistence_sq_error_sum,
            model_abs_error_on_persistence_sum,
            model_sq_error_on_persistence_sum,
            persistence_valid_sum,
            target_std.to(device),
        )
    estimated_persistence_metrics = None
    if estimated_persistence_batch_count.item() > 0:
        estimated_persistence_metrics = _baseline_error_metrics(
            estimated_persistence_abs_error_sum,
            estimated_persistence_sq_error_sum,
            estimated_persistence_valid_sum,
            target_std.to(device),
        )
        estimated_persistence_metrics = _add_paired_baseline_metrics(
            estimated_persistence_metrics,
            estimated_persistence_abs_error_sum,
            estimated_persistence_sq_error_sum,
            model_abs_error_on_estimated_persistence_sum,
            model_sq_error_on_estimated_persistence_sum,
            estimated_persistence_valid_sum,
            target_std.to(device),
        )
    loss_value = (total_loss_sum / total_count.clamp_min(1.0)).item()
    if not multi_horizon:
        return {
            "loss": loss_value,
            **_serialize_tensor_metrics(metrics),
            "valid_count": valid_sum.detach().cpu().tolist(),
            "baselines": {
                "train_mean": _serialize_tensor_metrics(train_mean_metrics),
                "persistence": (
                    _serialize_tensor_metrics(persistence_metrics)
                    if persistence_metrics is not None
                    else None
                ),
                "estimated_persistence": (
                    _serialize_tensor_metrics(estimated_persistence_metrics)
                    if estimated_persistence_metrics is not None
                    else None
                ),
            },
        }

    per_horizon = {}
    for horizon_index, horizon_seconds in enumerate(future_horizons_seconds):
        horizon_key = f"{horizon_seconds:g}"
        per_horizon[horizon_key] = {
            "horizon_seconds": horizon_seconds,
            **_select_horizon_metrics(metrics, horizon_index),
            "valid_count": valid_sum[horizon_index].detach().cpu().tolist(),
            "baselines": {
                "train_mean": _select_horizon_metrics(
                    train_mean_metrics,
                    horizon_index,
                ),
                "persistence": (
                    _select_horizon_metrics(persistence_metrics, horizon_index)
                    if persistence_metrics is not None
                    else None
                ),
                "estimated_persistence": (
                    _select_horizon_metrics(estimated_persistence_metrics, horizon_index)
                    if estimated_persistence_metrics is not None
                    else None
                ),
            },
        }

    endpoint_index = len(future_horizons_seconds) - 1
    return {
        "loss": loss_value,
        **_select_horizon_metrics(metrics, endpoint_index),
        "valid_count": valid_sum[endpoint_index].detach().cpu().tolist(),
        "baselines": {
            "train_mean": _select_horizon_metrics(
                train_mean_metrics,
                endpoint_index,
            ),
                "persistence": (
                    _select_horizon_metrics(persistence_metrics, endpoint_index)
                    if persistence_metrics is not None
                    else None
                ),
                "estimated_persistence": (
                    _select_horizon_metrics(estimated_persistence_metrics, endpoint_index)
                    if estimated_persistence_metrics is not None
                    else None
                ),
        },
        "top_level_metric_horizon_seconds": future_horizons_seconds[endpoint_index],
        "future_horizons_seconds": future_horizons_seconds,
        "horizon_macro": {
            **_macro_horizon_metrics(metrics),
            "baselines": {
                "train_mean": _macro_horizon_metrics(train_mean_metrics),
                "persistence": (
                    _macro_horizon_metrics(persistence_metrics)
                    if persistence_metrics is not None
                    else None
                ),
                "estimated_persistence": (
                    _macro_horizon_metrics(estimated_persistence_metrics)
                    if estimated_persistence_metrics is not None
                    else None
                ),
            },
        },
        "per_horizon": per_horizon,
    }


def save_checkpoint(path, encoder_checkpoint, probe, optimizer, epoch, args, stats, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_checkpoint": encoder_checkpoint,
            "probe": unwrap_model(probe).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "target_mean": stats[0].cpu(),
            "target_std": stats[1].cpu(),
            "metrics": metrics,
        },
        path,
    )
