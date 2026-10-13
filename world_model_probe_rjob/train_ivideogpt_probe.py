#!/usr/bin/env python3
"""Train the existing physical probe on frozen iVideoGPT causal states.

This entry point is deliberately separate from ``train_world_model_probe.py``.
It imports the established dataset/probe engine read-only and replaces four
module globals only inside this Python process: argument parsing, architecture,
checkpoint reporting, and input preprocessing.
"""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch.distributed as dist


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parent
SHARED_PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
for path in (str(RUN_DIR), str(REPO_ROOT), str(SHARED_PROBE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import src.datasets.utils.video.transforms as video_transforms  # noqa: E402
import src.datasets.utils.video.volume_transforms as volume_transforms  # noqa: E402
import train_covla_probe as shared_runner  # noqa: E402

from encoder_adapters.ivideogpt import IVideoGPTFrozenAdapter  # noqa: E402
from ivideogpt_config import parse_args  # noqa: E402


INPUT_FRAMES = 64
INPUT_SIZE = 64
FEATURE_DIM = 1024
NUM_HEADS = 16


class _LetterboxClip:
    """Resize each frame without changing aspect ratio, then pad to a square."""

    def __init__(self, size, fill_value=0):
        self.size = int(size)
        self.fill_value = int(fill_value)

    def __call__(self, clip):
        if not clip:
            return clip
        output = []
        for frame in clip:
            if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                raise TypeError(
                    "iVideoGPT letterbox preprocessing expects HWC numpy frames"
                )
            height, width = frame.shape[:2]
            scale = min(self.size / float(width), self.size / float(height))
            new_width = max(1, min(self.size, int(round(width * scale))))
            new_height = max(1, min(self.size, int(round(height * scale))))
            resized = cv2.resize(
                frame,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
            canvas = np.full(
                (self.size, self.size, frame.shape[2]),
                self.fill_value,
                dtype=frame.dtype,
            )
            top = (self.size - new_height) // 2
            left = (self.size - new_width) // 2
            canvas[top : top + new_height, left : left + new_width] = resized
            output.append(canvas)
        return output


def build_ivideogpt_transform(img_size, input_preprocess="direct"):
    """Build checkpoint-compatible RGB [0,1] preprocessing.

    ``direct`` is the registered baseline and intentionally stretches every
    source frame to a square.  The two alternatives are diagnostic policies
    for views whose native aspect ratio differs substantially from 1:1.
    """

    if int(img_size) != INPUT_SIZE:
        raise ValueError(f"iVideoGPT transform requires size {INPUT_SIZE}, got {img_size}")
    input_preprocess = str(input_preprocess)
    if input_preprocess == "direct":
        spatial = video_transforms.Resize(
            (INPUT_SIZE, INPUT_SIZE), interpolation="bilinear"
        )
    elif input_preprocess == "center_crop":
        spatial = video_transforms.Compose(
            [
                video_transforms.Resize(INPUT_SIZE, interpolation="bilinear"),
                video_transforms.CenterCrop((INPUT_SIZE, INPUT_SIZE)),
            ]
        )
    elif input_preprocess == "letterbox":
        spatial = _LetterboxClip(INPUT_SIZE)
    else:
        raise ValueError(
            f"Unknown iVideoGPT input_preprocess={input_preprocess!r}; "
            "expected direct, center_crop, or letterbox"
        )
    return video_transforms.Compose(
        [
            spatial,
            volume_transforms.ClipToTensor(),
        ]
    )


def build_adapter(args):
    adapter = IVideoGPTFrozenAdapter(
        checkpoint_path=args.encoder_checkpoint,
        input_frames=INPUT_FRAMES,
        input_size=INPUT_SIZE,
        target_group_size=args.tubelet_size,
        window_frames=args.window_frames,
        context_frames=args.context_frames,
        encoder_dtype=args.encoder_dtype,
        attention_implementation=args.attention_implementation,
    )
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.load_report.update(
        {
            "training_mode": "frozen_tokenizer_and_transformer_probe_only",
            "trainable_encoder_parameters": sum(
                parameter.numel()
                for parameter in adapter.parameters()
                if parameter.requires_grad
            ),
            "input_transform": {
                "resize": [INPUT_SIZE, INPUT_SIZE],
                "resize_policy": args.input_preprocess,
                "tensor_range": "[0, 1]",
                "normalize": False,
            },
            "shared_probe_training_amp": (
                "bfloat16" if args.use_bfloat16 else "float16"
            ),
            "encoder_autocast_override": "float32",
        }
    )
    return adapter


def _run_adapter_only_check(args):
    adapter = build_adapter(args)
    report = {
        "status": "ok",
        **adapter.load_report,
        "parameter_count_total": sum(
            parameter.numel() for parameter in adapter.parameters()
        ),
        "trainable_after_runner_freeze": sum(
            parameter.numel()
            for parameter in adapter.parameters()
            if parameter.requires_grad
        ),
        "note": "Formal training uses this identical construction path.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main(argv=None):
    args = parse_args(argv)
    if args.adapter_only_check:
        _run_adapter_only_check(args)
        return

    def adapter_builder(img_size, num_frames, **unused_kwargs):
        normalized_size = (
            tuple(img_size)
            if isinstance(img_size, (tuple, list))
            else (img_size, img_size)
        )
        if normalized_size != (INPUT_SIZE, INPUT_SIZE) or num_frames != INPUT_FRAMES:
            raise ValueError(
                "Injected iVideoGPT architecture mismatch: "
                f"img_size={normalized_size}, num_frames={num_frames}"
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
        expected = Path(args.encoder_checkpoint).resolve()
        supplied = Path(checkpoint_path).resolve()
        if supplied != expected:
            raise RuntimeError(
                f"Checkpoint changed after adapter construction: {supplied} != {expected}"
            )
        if checkpoint_key != "huggingface":
            raise RuntimeError(
                f"Unexpected iVideoGPT checkpoint_key={checkpoint_key!r}"
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "adapter": model.load_report,
                "config": args.config,
                "base_config": args.base_config,
                "shared_probe_runner": str(
                    SHARED_PROBE_DIR / "train_covla_probe.py"
                ),
                "shared_probe_code_modified_by_this_runner": False,
                "existing_internvideo_runner_modified": False,
            }
            with (output_dir / "ivideogpt_manifest.json").open("w") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return model.load_report

    # These assignments affect this process only. No imported file is edited.
    shared_runner.parse_args = lambda: args
    shared_runner.ARCH_SPECS = {"ivideogpt_oxe_64_medium": architecture}
    shared_runner.load_encoder = report_preloaded_encoder
    shared_runner.build_transform = lambda img_size: build_ivideogpt_transform(
        img_size, args.input_preprocess
    )
    shared_runner.main()


if __name__ == "__main__":
    main()
