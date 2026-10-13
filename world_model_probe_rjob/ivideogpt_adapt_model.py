"""Trainable iVideoGPT causal transformer plus optional Time auxiliary head."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.time_jepa.physical_aux import build_physical_head, masked_smooth_l1_loss
from world_model_probe_rjob.encoder_adapters.ivideogpt import (
    PINNED_REVISION,
    _snapshot_revision,
    build_token_layout,
    extract_boundary_states,
    resolve_snapshot_root,
    split_native_windows,
)


def load_trainable_components(
    base_snapshot: str,
    attention_implementation: str = "sdpa",
    gradient_checkpointing: bool = True,
):
    """Load the official tokenizer frozen and the full causal LM trainable."""

    snapshot_root = resolve_snapshot_root(base_snapshot)
    revision = _snapshot_revision(snapshot_root)
    if revision is not None and revision != PINNED_REVISION:
        raise RuntimeError(
            f"Snapshot revision mismatch: {revision} != {PINNED_REVISION}"
        )

    from ivideogpt.vq_model import CompressiveVQModel
    from transformers import AutoModelForCausalLM

    tokenizer = CompressiveVQModel.from_pretrained(
        str(snapshot_root),
        subfolder="tokenizer",
        local_files_only=True,
        low_cpu_mem_usage=False,
    ).float()
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    import transformers

    kwargs = {
        "subfolder": "transformer",
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attention_implementation,
    }
    major_version = int(transformers.__version__.split(".", 1)[0])
    kwargs["dtype" if major_version >= 5 else "torch_dtype"] = torch.float32
    transformer = AutoModelForCausalLM.from_pretrained(str(snapshot_root), **kwargs)
    transformer.config.use_cache = False
    transformer.train()
    transformer.requires_grad_(True)
    if gradient_checkpointing:
        transformer.gradient_checkpointing_enable()
    return tokenizer, transformer, snapshot_root


def append_terminal_sdf(tokens, labels, sdf_token_id):
    terminal = torch.full(
        (tokens.shape[0], 1),
        int(sdf_token_id),
        dtype=tokens.dtype,
        device=tokens.device,
    )
    ignored = torch.full(
        (labels.shape[0], 1), -100, dtype=labels.dtype, device=labels.device
    )
    return torch.cat((tokens, terminal), dim=1), torch.cat((labels, ignored), dim=1)


class IVideoGPTAdaptModel(nn.Module):
    def __init__(
        self,
        transformer,
        mode: str,
        num_physical_outputs: int = 0,
        physical_head_type: str = "conv1x1",
        smooth_l1_beta: float = 0.5,
    ):
        super().__init__()
        if mode not in ("video_only", "time"):
            raise ValueError(mode)
        self.transformer = transformer
        self.mode = mode
        self.layout = build_token_layout(
            window_frames=16,
            context_frames=2,
            target_group_size=2,
        )
        self.num_windows = 4
        self.sdf_token_id = 16385
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.register_buffer(
            "boundary_indices",
            torch.tensor(self.layout.boundary_indices, dtype=torch.long),
            persistent=False,
        )
        self.physical_head = None
        if mode == "time":
            if num_physical_outputs <= 0:
                raise ValueError("Time mode needs positive num_physical_outputs")
            self.physical_head = build_physical_head(
                head_type=physical_head_type,
                embed_dim=1024,
                num_outputs=int(num_physical_outputs),
                num_frames=64,
                tubelet_size=2,
            )

    def forward(
        self,
        tokens,
        labels,
        clip_batch_size: int,
        target=None,
        valid=None,
        target_mean=None,
        target_std=None,
        lambda_time: float = 0.0,
    ):
        if tokens.shape != labels.shape:
            raise ValueError(f"tokens/labels mismatch: {tokens.shape} vs {labels.shape}")
        if tokens.shape[1] != self.layout.sequence_tokens_with_terminal_sdf:
            raise ValueError(
                f"Expected token length {self.layout.sequence_tokens_with_terminal_sdf}, "
                f"got {tokens.shape[1]}"
            )
        outputs = self.transformer.model(
            input_ids=tokens,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = self.transformer.lm_head(hidden)
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        loss_video = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        states = extract_boundary_states(
            hidden,
            batch_size=int(clip_batch_size),
            num_windows=self.num_windows,
            layout=self.layout,
        )
        loss_physical = loss_video.new_zeros(())
        predictions = None
        if self.mode == "time":
            if any(value is None for value in (target, valid, target_mean, target_std)):
                raise ValueError("Time mode requires target, valid, mean, and std")
            normalized_target = (target.float() - target_mean) / target_std
            predictions = self.physical_head(states)
            loss_physical = masked_smooth_l1_loss(
                predictions,
                normalized_target,
                valid,
                beta=self.smooth_l1_beta,
            )
        total = loss_video + float(lambda_time) * loss_physical
        return {
            "loss": total,
            "loss_video": loss_video.detach(),
            "loss_physical": loss_physical.detach(),
            "states": states,
            "predictions": predictions,
        }


@torch.no_grad()
def tokenize_clips(tokenizer, videos, layout, sdf_token_id=16385):
    """Tokenize [B,C,64,64,64] into four native causal sequences per clip."""

    if videos.ndim != 5 or tuple(videos.shape[1:]) != (3, 64, 64, 64):
        raise ValueError(f"Expected videos [B,3,64,64,64], got {tuple(videos.shape)}")
    windows = split_native_windows(videos.float(), layout.window_frames)
    device_type = videos.device.type
    autocast = (
        torch.amp.autocast(device_type=device_type, enabled=False)
        if device_type in ("cpu", "cuda")
        else nullcontext()
    )
    with autocast:
        tokens, labels = tokenizer.tokenize(
            windows, context_length=layout.context_frames
        )
    if tokens.shape[1] != layout.native_sequence_tokens:
        raise RuntimeError(
            f"Tokenizer produced {tokens.shape[1]} tokens, expected {layout.native_sequence_tokens}"
        )
    boundary_indices = torch.as_tensor(
        layout.boundary_indices[:-1], device=tokens.device, dtype=torch.long
    )
    if not torch.all(tokens.index_select(1, boundary_indices) == int(sdf_token_id)):
        raise RuntimeError("Tokenizer SDF markers do not match registered layout")
    return append_terminal_sdf(tokens, labels, sdf_token_id)


def optimizer_parameter_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]

