#!/usr/bin/env python3
"""Train a fresh physical probe on a frozen adapted iVideoGPT transformer."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch.distributed as dist


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parent
SHARED_PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
for path in (str(RUN_DIR), str(REPO_ROOT), str(SHARED_PROBE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import train_covla_probe as shared_runner  # noqa: E402

from encoder_adapters.ivideogpt_adapted import IVideoGPTAdaptedAdapter  # noqa: E402
from ivideogpt_adapted_probe_config import parse_args  # noqa: E402
from train_ivideogpt_probe import (  # noqa: E402
    FEATURE_DIM,
    INPUT_FRAMES,
    INPUT_SIZE,
    NUM_HEADS,
    build_ivideogpt_transform,
)


def build_adapter(args):
    adapter = IVideoGPTAdaptedAdapter(
        base_checkpoint_path=args.encoder_checkpoint,
        adapted_transformer_checkpoint=args.adapted_transformer_checkpoint,
        expected_adaptation_mode=args.adaptation_mode,
        expected_adaptation_epoch=args.adaptation_epoch,
        input_frames=INPUT_FRAMES,
        input_size=INPUT_SIZE,
        target_group_size=args.tubelet_size,
        window_frames=args.window_frames,
        context_frames=args.context_frames,
        encoder_dtype=args.encoder_dtype,
        attention_implementation=args.attention_implementation,
    )
    adapter.eval()
    adapter.requires_grad_(False)
    adapter.load_report.update(
        {
            "training_mode": "frozen_adapted_transformer_fresh_probe_only",
            "adaptation_mode": args.adaptation_mode,
            "adaptation_epoch": args.adaptation_epoch,
            "trainable_encoder_parameters": 0,
            "temporary_adaptation_head_loaded": False,
            "input_transform": {
                "resize": [INPUT_SIZE, INPUT_SIZE],
                "resize_policy": args.input_preprocess,
                "tensor_range": "[0, 1]",
                "normalize": False,
            },
        }
    )
    return adapter


def main(argv=None):
    args = parse_args(argv)
    if args.adapter_only_check:
        adapter = build_adapter(args)
        print(
            json.dumps(
                {
                    "status": "ok",
                    **adapter.load_report,
                    "trainable_after_runner_freeze": sum(
                        parameter.numel()
                        for parameter in adapter.parameters()
                        if parameter.requires_grad
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    def adapter_builder(img_size, num_frames, **_):
        normalized_size = (
            tuple(img_size) if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        )
        if normalized_size != (INPUT_SIZE, INPUT_SIZE) or num_frames != INPUT_FRAMES:
            raise ValueError(
                f"Adapted iVideoGPT architecture mismatch: {normalized_size}, {num_frames}"
            )
        return build_adapter(args)

    architecture = {
        "builder": adapter_builder,
        "legacy_builder": adapter_builder,
        "img_size": INPUT_SIZE,
        "num_frames": INPUT_FRAMES,
        "embed_dim": FEATURE_DIM,
        "num_heads": NUM_HEADS,
    }

    def report_preloaded_encoder(model, checkpoint_path, checkpoint_key="huggingface"):
        if Path(checkpoint_path).resolve() != Path(args.encoder_checkpoint).resolve():
            raise RuntimeError("Base tokenizer checkpoint changed after adapter construction")
        if checkpoint_key != "huggingface":
            raise RuntimeError(f"Unexpected checkpoint_key={checkpoint_key}")
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "ivideogpt_adapted_manifest.json").open("w") as handle:
                json.dump(
                    {
                        "adapter": model.load_report,
                        "config": args.config,
                        "base_config": args.base_config,
                        "fresh_probe": True,
                        "temporary_adaptation_head_reused": False,
                        "shared_probe_code_modified_by_this_runner": False,
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
        return model.load_report

    shared_runner.parse_args = lambda: args
    shared_runner.ARCH_SPECS = {"ivideogpt_oxe_64_medium": architecture}
    shared_runner.load_encoder = report_preloaded_encoder
    shared_runner.build_transform = lambda img_size: build_ivideogpt_transform(
        img_size, args.input_preprocess
    )
    shared_runner.main()


if __name__ == "__main__":
    main()
