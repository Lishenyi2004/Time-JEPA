"""Loss composition shared by PSG-Static-g-Port and PSG-Full-g-Port."""

from __future__ import annotations

import torch

from .static_only import masked_mse


def psg_vitg_forward(self, batch, stage, cfg):
    """Apply the authors' JEPA/SIGReg recipe and selected PSG grounding form."""

    context_length = int(cfg.wm.history_size)
    num_predictions = int(cfg.wm.num_preds)
    sequence_length = context_length + num_predictions
    sigreg_weight = float(cfg.loss.sigreg.weight)
    grounding_weight = float(cfg.loss.grounding.weight)
    grounding_variant = str(cfg.loss.grounding.variant)
    if grounding_variant != self.model.grounding_variant:
        raise RuntimeError(
            "Configured/model grounding variants differ: "
            f"{grounding_variant!r} vs {self.model.grounding_variant!r}"
        )

    model_batch = dict(batch)
    model_batch["action"] = torch.nan_to_num(model_batch["action"], 0.0)
    output = self.model.encode(model_batch)
    embeddings = output["emb"]
    action_embeddings = output["act_emb"]
    if embeddings.ndim != 3 or embeddings.shape[1] != sequence_length:
        raise ValueError(
            f"PSG expects embeddings [B,{sequence_length},D], got "
            f"{tuple(embeddings.shape)}"
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

    if grounding_weight <= 0:
        raise ValueError("Formal PSG-g rows require loss.grounding.weight > 0")
    if "observation" not in batch:
        raise KeyError("PSG grounding requires batch['observation']")
    observation = batch["observation"]
    observation_valid = batch.get("observation_valid")
    if observation_valid is not None:
        observation = observation.masked_fill(observation_valid <= 0, float("nan"))

    batch_size, time_steps, embedding_dim = embeddings.shape
    if grounding_variant == "static":
        state_prediction = self.model.grounding.state_head(
            embeddings.reshape(batch_size * time_steps, embedding_dim)
        ).reshape(batch_size, time_steps, -1)
        grounding_loss = masked_mse(
            state_prediction,
            observation,
            observation_valid,
        )
        output["grounding_static_loss"] = grounding_loss.detach()
    elif grounding_variant == "full":
        grounding_output = self.model.upstream_grounding_loss(
            self.model.grounding,
            embeddings,
            observation,
            state_idx=[int(index) for index in cfg.loss.grounding.state_idx],
            joint_idx=[int(index) for index in cfg.loss.grounding.joint_idx],
            vel_idx=[int(index) for index in cfg.loss.grounding.vel_idx],
        )
        grounding_loss = grounding_output["loss"]
        for key in ("static", "djoint", "velocity"):
            if key in grounding_output:
                output[f"grounding_{key}_loss"] = grounding_output[key]
    else:
        raise ValueError(f"Unknown PSG grounding variant: {grounding_variant}")

    total_loss = total_loss + grounding_weight * grounding_loss
    output["grounding_loss"] = grounding_loss.detach()
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
