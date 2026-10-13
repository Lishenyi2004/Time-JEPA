#!/usr/bin/env python3
"""Run an isolated world-model encoder with the existing physical probe stack.

No file in ``covla_probe_rjob`` is modified.  That directory is imported as a
read-only scientific protocol implementation (datasets, target construction,
normalization, probe heads, metrics and reporting), while encoder construction
and checkpoint loading are injected here.
"""

import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
for path in (str(REPO_ROOT), str(SHARED_PROBE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import train_covla_probe as shared_runner  # noqa: E402

from encoder_adapters import InternVideoNextAdapter, VideoMAEV2GAdapter  # noqa: E402
from world_model_config import parse_args  # noqa: E402


ARCHITECTURES = {
    "internvideo_next_l": {
        "input_frames": 64,
        "input_size": 224,
        "feature_dim": 1024,
        "num_heads": 16,
    },
    "videomaev2_g": {
        "input_frames": 64,
        "input_size": 224,
        "feature_dim": 1408,
        "num_heads": 16,
    },
}


def make_autocast_evaluate(evaluate_fn, amp_dtype):
    """Run the shared evaluation loop under the same AMP policy as training.

    The frozen encoders emit checkpoint-native FP16/BF16 features. The shared V-JEPA
    evaluator normally invokes its FP32 probe outside autocast, which makes a
    Conv1d fail with a feature/bias dtype mismatch. Keeping the whole shared
    evaluation call in autocast lets PyTorch choose a compatible convolution
    kernel without materializing the large patch-token feature tensor in FP32.
    """

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


def build_adapter(args):
    architecture = ARCHITECTURES[args.arch]
    common = {
        "checkpoint_path": args.encoder_checkpoint,
        "input_frames": architecture["input_frames"],
        "input_size": architecture["input_size"],
        "target_group_size": args.tubelet_size,
        "encoder_dtype": args.encoder_dtype,
    }
    if args.encoder_family == "internvideo_next":
        adapter = InternVideoNextAdapter(
            **common,
            kernel_mode=args.kernel_mode,
        )
    elif args.encoder_family == "videomaev2":
        adapter = VideoMAEV2GAdapter(
            **common,
            attention_implementation=args.attention_implementation,
            temporal_mode=args.videomae_temporal_mode,
            encoder_microbatch_size=args.encoder_microbatch_size,
        )
    else:  # Defensive: argparse and protocol validation should catch this first.
        raise ValueError(f"Unsupported encoder_family: {args.encoder_family}")
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.load_report["training_mode"] = "frozen_encoder_probe_only"
    adapter.load_report["probe_amp_dtype"] = (
        "bfloat16" if args.use_bfloat16 else "float16"
    )
    adapter.load_report["probe_evaluation_autocast"] = True
    adapter.load_report["trainable_encoder_parameters"] = sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )
    return adapter


def _run_adapter_only_check(args):
    adapter = build_adapter(args)
    report = {
        **adapter.load_report,
        "parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "trainable_after_runner_freeze": sum(
            parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
        ),
        "note": "The same build path is used by formal training before optimizer creation.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main(argv=None):
    args = parse_args(argv)
    if args.adapter_only_check:
        _run_adapter_only_check(args)
        return

    selected_architecture = ARCHITECTURES[args.arch]
    input_size = selected_architecture["input_size"]
    input_frames = selected_architecture["input_frames"]

    def adapter_builder(img_size, num_frames, **unused_kwargs):
        normalized_size = tuple(img_size) if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        if normalized_size != (input_size, input_size) or num_frames != input_frames:
            raise ValueError(
                f"Injected {args.arch} architecture mismatch: "
                f"img_size={normalized_size}, num_frames={num_frames}"
            )
        return build_adapter(args)

    architecture = {
        "builder": adapter_builder,
        "legacy_builder": adapter_builder,
        "img_size": input_size,
        "num_frames": input_frames,
        "embed_dim": selected_architecture["feature_dim"],
        "num_heads": selected_architecture["num_heads"],
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
                "encoder_family": args.encoder_family,
                "shared_probe_runner": str(SHARED_PROBE_DIR / "train_covla_probe.py"),
                "shared_probe_code_modified_by_this_runner": False,
            }
            with (output_dir / "world_model_manifest.json").open("w") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return model.load_report

    # The imported module remains unchanged on disk. These process-local globals
    # are the narrow injection boundary for its already-tested training engine.
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
