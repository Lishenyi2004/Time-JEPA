"""Shared frozen world-model adapters for physical-consistency probe banks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


WORLD_MODEL_BACKENDS = frozenset(
    {"ivideogpt", "internvideo_next", "videomaev2"}
)

ADAPTATION_METADATA_KEYS = (
    "adapted_transformer_checkpoint",
    "adaptation_mode",
    "adaptation_epoch",
    "adaptation_domain",
)


def validate_world_model_encoder_config(config: dict) -> bool:
    """Validate a registered world-model backend; return False if not handled."""

    encoder = config.get("encoder", {})
    sampling = config.get("sampling", {})
    backend = encoder.get("backend")
    if backend not in WORLD_MODEL_BACKENDS:
        return False

    if not encoder.get("checkpoint"):
        raise ValueError(f"{backend} requires encoder.checkpoint")

    if backend == "ivideogpt":
        if int(sampling.get("image_size", 0)) != 64:
            raise ValueError("iVideoGPT runs require sampling.image_size=64")
        if encoder.get("encoder_dtype", "float32") != "float32":
            raise ValueError("Registered iVideoGPT runs require encoder_dtype=float32")
        if int(encoder.get("window_frames", 16)) != 16:
            raise ValueError("Registered iVideoGPT runs require window_frames=16")
        if int(encoder.get("context_frames", 2)) != 2:
            raise ValueError("Registered iVideoGPT runs require context_frames=2")

        adapted_checkpoint = encoder.get("adapted_transformer_checkpoint")
        adaptation_mode = encoder.get("adaptation_mode")
        adaptation_epoch = encoder.get("adaptation_epoch")
        adaptation_domain = encoder.get("adaptation_domain")
        if adapted_checkpoint:
            if adaptation_mode not in {"time", "video_only"}:
                raise ValueError(
                    "Adapted iVideoGPT runs require adaptation_mode=time or video_only"
                )
            if (
                isinstance(adaptation_epoch, bool)
                or not isinstance(adaptation_epoch, int)
                or adaptation_epoch < 0
            ):
                raise ValueError(
                    "Adapted iVideoGPT runs require a non-negative adaptation_epoch"
                )
            if adaptation_domain not in {"egoexo", "agibot", "covla"}:
                raise ValueError(
                    "Adapted iVideoGPT runs require adaptation_domain to be "
                    "egoexo, agibot, or covla"
                )
        elif any(
            value is not None
            for value in (adaptation_mode, adaptation_epoch, adaptation_domain)
        ):
            raise ValueError(
                "Frozen iVideoGPT runs must not declare adaptation metadata without "
                "adapted_transformer_checkpoint"
            )
    elif backend == "internvideo_next":
        if int(sampling.get("image_size", 0)) != 224:
            raise ValueError("InternVideo-Next runs require sampling.image_size=224")
        if encoder.get("encoder_dtype", "float16") not in {
            "float16",
            "bfloat16",
            "float32",
        }:
            raise ValueError("Unsupported InternVideo-Next encoder_dtype")
        if encoder.get("kernel_mode", "auto") not in {
            "auto",
            "native",
            "portable",
        }:
            raise ValueError("Unsupported InternVideo-Next kernel_mode")
        if any(
            encoder.get(key) is not None
            for key in ADAPTATION_METADATA_KEYS
        ):
            raise ValueError("InternVideo-Next zero must use its frozen base checkpoint")
    else:
        if int(sampling.get("num_frames", 0)) != 64:
            raise ValueError("VideoMAE V2-g runs require sampling.num_frames=64")
        if int(sampling.get("image_size", 0)) != 224:
            raise ValueError("VideoMAE V2-g runs require sampling.image_size=224")
        if int(sampling.get("tubelet_size", 0)) != 2:
            raise ValueError("VideoMAE V2-g runs require sampling.tubelet_size=2")
        if encoder.get("checkpoint_key", "model") != "model":
            raise ValueError("VideoMAE V2-g requires encoder.checkpoint_key=model")
        if encoder.get("encoder_dtype", "float16") not in {
            "float16",
            "bfloat16",
            "float32",
        }:
            raise ValueError("Unsupported VideoMAE V2-g encoder_dtype")
        attention_implementation = encoder.get(
            "attention_implementation", "sdpa"
        )
        if attention_implementation not in {"sdpa", "eager"}:
            raise ValueError(
                "VideoMAE V2-g attention_implementation must be sdpa or eager"
            )
        temporal_mode = encoder.get("temporal_mode", "full")
        if temporal_mode not in {"full", "native_chunks"}:
            raise ValueError(
                "VideoMAE V2-g temporal_mode must be full or native_chunks"
            )
        if temporal_mode == "full" and attention_implementation != "sdpa":
            raise ValueError(
                "VideoMAE V2-g full temporal mode requires fused SDPA"
            )
        microbatch_size = encoder.get("encoder_microbatch_size", 1)
        if (
            isinstance(microbatch_size, bool)
            or not isinstance(microbatch_size, int)
            or microbatch_size <= 0
        ):
            raise ValueError(
                "VideoMAE V2-g encoder_microbatch_size must be a positive integer"
            )
        if any(
            encoder.get(key) is not None
            for key in ADAPTATION_METADATA_KEYS
        ):
            raise ValueError(
                "VideoMAE V2-g zero must use its frozen clean-pretraining checkpoint"
            )
    return True


def build_world_model_encoder(config: dict):
    """Build one registered world-model adapter and return encoder/dim/report."""

    if not validate_world_model_encoder_config(config):
        raise ValueError(
            f"Unsupported shared world-model backend: {config.get('encoder', {}).get('backend')}"
        )

    encoder_config = config["encoder"]
    sampling = config["sampling"]
    backend = encoder_config["backend"]
    input_frames = int(sampling["num_frames"])
    input_size = int(sampling["image_size"])
    target_group_size = int(sampling["tubelet_size"])

    if backend == "ivideogpt":
        from world_model_probe_rjob.encoder_adapters.ivideogpt import (
            IVideoGPTFrozenAdapter,
        )
        from world_model_probe_rjob.encoder_adapters.ivideogpt_adapted import (
            IVideoGPTAdaptedAdapter,
        )

        base_checkpoint = Path(encoder_config["checkpoint"]).expanduser().resolve()
        common_kwargs = {
            "input_frames": input_frames,
            "input_size": input_size,
            "target_group_size": target_group_size,
            "window_frames": int(encoder_config.get("window_frames", 16)),
            "context_frames": int(encoder_config.get("context_frames", 2)),
            "encoder_dtype": encoder_config.get("encoder_dtype", "float32"),
            "attention_implementation": encoder_config.get(
                "attention_implementation", "sdpa"
            ),
        }
        adapted_value = encoder_config.get("adapted_transformer_checkpoint")
        if adapted_value:
            # Preserve the lexical checkpoint directory so manifest lookup also
            # remains correct when a future retained checkpoint uses symlinks.
            adapted_checkpoint = Path(
                os.path.abspath(Path(adapted_value).expanduser())
            )
            encoder = IVideoGPTAdaptedAdapter(
                base_checkpoint_path=base_checkpoint,
                adapted_transformer_checkpoint=adapted_checkpoint,
                expected_adaptation_mode=encoder_config["adaptation_mode"],
                expected_adaptation_epoch=int(encoder_config["adaptation_epoch"]),
                expected_adaptation_domain=encoder_config["adaptation_domain"],
                **common_kwargs,
            )
            selected_checkpoint = adapted_checkpoint
            variant = "adapted"
        else:
            encoder = IVideoGPTFrozenAdapter(
                checkpoint_path=base_checkpoint,
                **common_kwargs,
            )
            selected_checkpoint = base_checkpoint
            variant = "frozen"

        expected_tokens = input_frames // target_group_size
        if int(encoder.spec.output_tokens) != expected_tokens:
            raise RuntimeError(
                "iVideoGPT temporal output mismatch: "
                f"{encoder.spec.output_tokens} != {expected_tokens}"
            )
        embed_dim = int(encoder.spec.feature_dim)
        report = dict(encoder.load_report)
        report.update(
            {
                "backend": backend,
                "variant": variant,
                "checkpoint": str(selected_checkpoint),
                "base_checkpoint": str(base_checkpoint),
                "checkpoint_size_bytes": selected_checkpoint.stat().st_size,
            }
        )
        return encoder, embed_dim, report

    if backend == "videomaev2":
        from world_model_probe_rjob.encoder_adapters.videomaev2 import (
            VideoMAEV2GAdapter,
        )

        checkpoint = Path(encoder_config["checkpoint"]).expanduser().resolve()
        encoder = VideoMAEV2GAdapter(
            checkpoint_path=checkpoint,
            input_frames=input_frames,
            input_size=input_size,
            target_group_size=target_group_size,
            encoder_dtype=encoder_config.get("encoder_dtype", "float16"),
            attention_implementation=encoder_config.get(
                "attention_implementation", "sdpa"
            ),
            temporal_mode=encoder_config.get("temporal_mode", "full"),
            encoder_microbatch_size=int(
                encoder_config.get("encoder_microbatch_size", 1)
            ),
        )
        temporal_tokens = input_frames // target_group_size
        if int(encoder.spec.output_tokens) % temporal_tokens:
            raise RuntimeError(
                "VideoMAE V2-g output tokens are not divisible by the physical "
                f"time grid: {encoder.spec.output_tokens} % {temporal_tokens}"
            )
        embed_dim = int(encoder.spec.feature_dim)
        report = dict(encoder.load_report)
        checkpoint_validation = report.pop("checkpoint", None)
        report.update(
            {
                "backend": backend,
                "variant": "frozen",
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_validation": checkpoint_validation,
            }
        )
        return encoder, embed_dim, report

    from world_model_probe_rjob.encoder_adapters.internvideo_next import (
        InternVideoNextAdapter,
    )

    checkpoint = Path(encoder_config["checkpoint"]).expanduser().resolve()
    encoder = InternVideoNextAdapter(
        checkpoint_path=checkpoint,
        input_frames=input_frames,
        input_size=input_size,
        target_group_size=target_group_size,
        encoder_dtype=encoder_config.get("encoder_dtype", "float16"),
        kernel_mode=encoder_config.get("kernel_mode", "auto"),
    )
    temporal_tokens = input_frames // target_group_size
    if int(encoder.spec.output_tokens) % temporal_tokens:
        raise RuntimeError(
            "InternVideo-Next output tokens are not divisible by the physical "
            f"time grid: {encoder.spec.output_tokens} % {temporal_tokens}"
        )
    embed_dim = int(encoder.spec.feature_dim)
    report = dict(encoder.load_report)
    report.update(
        {
            "backend": backend,
            "variant": "frozen",
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        }
    )
    return encoder, embed_dim, report


def build_physics_input_transform(
    config: dict,
    default_builder: Callable[[int], object],
):
    """Select exact checkpoint preprocessing while preserving legacy builders."""

    image_size = int(config["sampling"]["image_size"])
    if config["encoder"]["backend"] == "ivideogpt":
        from world_model_probe_rjob.train_ivideogpt_probe import (
            build_ivideogpt_transform,
        )

        return build_ivideogpt_transform(image_size)
    return default_builder(image_size)


__all__ = [
    "WORLD_MODEL_BACKENDS",
    "build_physics_input_transform",
    "build_world_model_encoder",
    "validate_world_model_encoder_config",
]
