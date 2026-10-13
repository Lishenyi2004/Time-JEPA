"""Paper-faithful PSG-JEPA ablation retaining only single-latent grounding."""

from __future__ import annotations

import torch


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE over valid finite targets, matching upstream behavior on finite data."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"Static prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    finite = torch.isfinite(target)
    if valid is not None:
        if valid.shape != target.shape:
            raise ValueError(f"Static valid-mask shape mismatch: {valid.shape} vs {target.shape}")
        finite = finite & (valid > 0)
    safe_target = torch.where(finite, target, torch.zeros_like(target))
    squared_error = (prediction - safe_target).square()
    return (squared_error * finite).sum() / finite.sum().clamp_min(1)


def static_only_forward(self, batch, stage, cfg):
    """Official LeWM forward objective plus PSG's individual-latent state head."""

    context_length = int(cfg.wm.history_size)
    num_predictions = int(cfg.wm.num_preds)
    sigreg_weight = float(cfg.loss.sigreg.weight)
    grounding_weight = float(cfg.loss.grounding.weight)

    model_batch = dict(batch)
    model_batch["action"] = torch.nan_to_num(model_batch["action"], 0.0)
    output = self.model.encode(model_batch)
    embeddings = output["emb"]
    action_embeddings = output["act_emb"]

    sequence_length = context_length + num_predictions
    if embeddings.shape[1] != sequence_length:
        raise ValueError(
            f"PSG expects {sequence_length} frames, got embeddings {embeddings.shape}"
        )
    context_embeddings = embeddings[:, :context_length]
    context_actions = action_embeddings[:, :context_length]
    target_embeddings = embeddings[:, num_predictions:sequence_length]
    predicted_embeddings = self.model.predict(context_embeddings, context_actions)

    prediction_loss = (predicted_embeddings - target_embeddings).square().mean()
    sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
    total_loss = prediction_loss + sigreg_weight * sigreg_loss
    output["pred_loss"] = prediction_loss
    output["sigreg_loss"] = sigreg_loss

    if grounding_weight > 0:
        if self.model.grounding is None or not hasattr(
            self.model.grounding,
            "state_head",
        ):
            raise RuntimeError("Static grounding is enabled but the official state head is absent")
        if "observation" not in batch:
            raise KeyError("Static grounding requires batch['observation']")
        observation = batch["observation"]
        batch_size, time_steps, feature_dim = embeddings.shape
        state_prediction = self.model.grounding.state_head(
            embeddings.reshape(batch_size * time_steps, feature_dim)
        ).reshape(batch_size, time_steps, -1)
        static_loss = masked_mse(
            state_prediction,
            observation,
            batch.get("observation_valid"),
        )
        total_loss = total_loss + grounding_weight * static_loss
        output["grounding_static_loss"] = static_loss
        output["grounding_loss"] = static_loss

    output["loss"] = total_loss
    metrics = {
        f"{stage}/{key}": value.detach()
        for key, value in output.items()
        if "loss" in key and torch.is_tensor(value) and value.ndim == 0
    }
    if hasattr(self, "log_dict"):
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=int(batch["pixels"].shape[0]),
        )
    return output
