#!/usr/bin/env python3
"""Run PSG-JEPA/LeWM with the existing frozen physical-probe protocol."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
for path in (str(REPO_ROOT), str(SHARED_PROBE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import train_covla_probe as shared_runner  # noqa: E402

from external_baselines.psg_jepa.probe_adapter import (  # noqa: E402
    PSG_FEATURE_DIM,
    PSG_INPUT_FRAMES,
    PSG_INPUT_SIZE,
    PSGJEPAFeatureAdapter,
)
from external_baselines.psg_jepa.probe_config import parse_args  # noqa: E402
from external_baselines.psg_jepa.vitg_probe_adapter import (  # noqa: E402
    PSG_VITG_FEATURE_DIM,
    PSG_VITG_INPUT_FRAMES,
    PSG_VITG_INPUT_SIZE,
    PSGJEPAViTGFeatureAdapter,
)


ARCHITECTURES = {
    "psg_jepa_vit_tiny": {
        "img_size": PSG_INPUT_SIZE,
        "num_frames": PSG_INPUT_FRAMES,
        "embed_dim": PSG_FEATURE_DIM,
        "num_heads": 3,
    },
    "psg_jepa_vitg": {
        "img_size": PSG_VITG_INPUT_SIZE,
        "num_frames": PSG_VITG_INPUT_FRAMES,
        "embed_dim": PSG_VITG_FEATURE_DIM,
        "num_heads": 22,
    },
}


def make_autocast_evaluate(evaluate_fn, amp_dtype):
    def evaluate_with_autocast(
        encoder,
        probe,
        loader,
        device,
        target_mean,
        target_std,
        **kwargs,
    ):
        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=amp_dtype,
        ):
            return evaluate_fn(
                encoder,
                probe,
                loader,
                device,
                target_mean,
                target_std,
                **kwargs,
            )

    return evaluate_with_autocast


def build_adapter(args, *, retain_state_head=False):
    if args.arch == "psg_jepa_vit_tiny":
        adapter = PSGJEPAFeatureAdapter(
            args.encoder_checkpoint,
            upstream_root=args.upstream_root,
            expected_variant=args.training_variant,
            frame_microbatch_size=args.frame_microbatch_size,
            retain_state_head=retain_state_head,
        )
    elif args.arch == "psg_jepa_vitg":
        if retain_state_head:
            raise ValueError(
                "ViT-g formal probes exclude the training-only grounding head"
            )
        adapter = PSGJEPAViTGFeatureAdapter(
            args.encoder_checkpoint,
            upstream_root=args.upstream_root,
            expected_variant=args.training_variant,
            frame_microbatch_size=args.frame_microbatch_size,
        )
    else:  # pragma: no cover - guarded by configuration validation
        raise ValueError(f"Unsupported PSG probe architecture: {args.arch!r}")
    adapter.load_report.update(
        {
            "training_mode": "frozen_encoder_projector_probe_only",
            "probe_amp_dtype": "bfloat16" if args.use_bfloat16 else "float16",
            "trainable_encoder_parameters": sum(
                parameter.numel()
                for parameter in adapter.parameters()
                if parameter.requires_grad
            ),
        }
    )
    return adapter


def _adapter_only_check(args):
    retain_state_head = (
        args.arch == "psg_jepa_vit_tiny"
        and args.training_variant == "psg_static"
    )
    adapter = build_adapter(args, retain_state_head=retain_state_head)
    report = {
        **adapter.load_report,
        "retained_parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main(argv=None):
    args = parse_args(argv)
    if args.adapter_only_check:
        _adapter_only_check(args)
        return

    contract = ARCHITECTURES[args.arch]

    def adapter_builder(img_size, num_frames, **unused_kwargs):
        normalized_size = (
            tuple(img_size) if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        )
        expected_size = int(contract["img_size"])
        if normalized_size != (expected_size, expected_size):
            raise ValueError(f"Injected PSG image size mismatch: {normalized_size}")
        if int(num_frames) != int(contract["num_frames"]):
            raise ValueError(f"Injected PSG frame count mismatch: {num_frames}")
        return build_adapter(args)

    architecture = {
        "builder": adapter_builder,
        "legacy_builder": adapter_builder,
        **contract,
    }

    def report_preloaded_encoder(model, checkpoint_path, checkpoint_key=None):
        expected = Path(args.encoder_checkpoint).resolve()
        supplied = Path(checkpoint_path).resolve()
        if supplied != expected:
            raise RuntimeError(f"Checkpoint changed after adapter construction: {supplied} != {expected}")
        if checkpoint_key != args.checkpoint_key:
            raise RuntimeError(
                f"Checkpoint key changed after validation: {checkpoint_key} != {args.checkpoint_key}"
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "adapter": model.load_report,
                "config": args.config,
                "base_config": args.base_config,
                "arch": args.arch,
                "training_variant": args.training_variant,
                "shared_probe_runner": str(SHARED_PROBE_DIR / "train_covla_probe.py"),
                "shared_probe_code_modified_by_this_runner": False,
            }
            with (output_dir / "psg_jepa_probe_manifest.json").open("w") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        return model.load_report

    amp_dtype = torch.bfloat16 if args.use_bfloat16 else torch.float16
    shared_runner.evaluate = make_autocast_evaluate(
        shared_runner.evaluate,
        amp_dtype=amp_dtype,
    )
    shared_runner.parse_args = lambda: args
    shared_runner.ARCH_SPECS = {args.arch: architecture}
    shared_runner.load_encoder = report_preloaded_encoder
    shared_runner.main()


if __name__ == "__main__":
    main()
