from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from physion_ocp_rjob.config import load_config, protocol_digest
from physion_ocp_rjob.dataset import inspect_dataset
from physion_ocp_rjob.model import (
    build_empty_target_encoder,
    compare_state_shapes,
    feature_dimension,
)


REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "PyYAML": "yaml",
    "numpy": "numpy",
    "h5py": "h5py",
    "decord": "decord",
    "Pillow": "PIL",
    "accelerate": "accelerate",
    "einops": "einops",
    "timm": "timm",
}
OPTIONAL_RELEASED_EVALUATOR_PACKAGES = {
    "scikit-learn": "sklearn",
    "imageio-ffmpeg": "imageio_ffmpeg",
}


def _package_versions(packages: dict[str, str], *, required: bool) -> dict[str, Any]:
    report: dict[str, Any] = {}
    failures: list[str] = []
    for distribution, module_name in packages.items():
        try:
            importlib.import_module(module_name)
            report[distribution] = {
                "available": True,
                "version": importlib.metadata.version(distribution),
            }
        except Exception as error:
            report[distribution] = {"available": False, "error": str(error)}
            failures.append(f"{distribution}/{module_name}: {error}")
    if required and failures:
        raise RuntimeError("Required dependency check failed: " + "; ".join(failures))
    return report


def check_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["checkpoint"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    target_encoder = build_empty_target_encoder(
        config["model"], int(config["extraction"]["num_frames"])
    )
    with FakeTensorMode():
        try:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except RuntimeError:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "target_encoder" not in checkpoint:
        raise KeyError(f"Checkpoint {path} has no target_encoder section")
    shape_report = compare_state_shapes(target_encoder, checkpoint["target_encoder"])
    if any(shape_report.values()):
        raise RuntimeError(
            "target_encoder shape/key mismatch: "
            + json.dumps(shape_report, ensure_ascii=False, default=str)[:12000]
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "epoch": checkpoint.get("epoch"),
        "target_encoder_tensors": len(checkpoint["target_encoder"]),
        "exact_match": True,
    }


def _storage_estimate(config: dict[str, Any]) -> dict[str, Any]:
    dimension = feature_dimension(config)
    dtype_bytes = 2 if config["extraction"]["feature_dtype"] == "float16" else 4
    extraction = config["extraction"]
    full_counts = {"train": 5608, "test": 1035}
    counts: dict[str, int] = {}
    for task in extraction["tasks"]:
        split = task["split"]
        cap = extraction.get(
            "max_train_samples" if split == "train" else "max_test_samples"
        )
        counts[f"{split}_{task['temporal_order']}"] = (
            full_counts[split] if cap is None else min(int(cap), full_counts[split])
        )
    total_rows = sum(counts.values())
    total_bytes = total_rows * dimension * dtype_bytes
    return {
        "aggregation": config["extraction"]["aggregation"],
        "feature_dimension": dimension,
        "feature_dtype_bytes": dtype_bytes,
        "rows_by_task": counts,
        "protocol_rows": total_rows,
        "estimated_feature_bytes": total_bytes,
        "estimated_feature_gib": total_bytes / (1024**3),
    }


def run_checks(
    config: dict[str, Any],
    *,
    require_cuda: bool,
    skip_checkpoint: bool,
    skip_decode: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "required_packages": _package_versions(REQUIRED_PACKAGES, required=True),
        "released_evaluator_optional_packages": _package_versions(
            OPTIONAL_RELEASED_EVALUATOR_PACKAGES, required=False
        ),
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "torch_cuda": torch.version.cuda,
        },
        "protocol_digest": protocol_digest(config),
        "storage_estimate": _storage_estimate(config),
        "implementation": {
            "video_decoder": "decord",
            "probe": "PyTorch standardized logistic regression",
            "requires_scikit_learn": False,
            "requires_imageio_ffmpeg": False,
            "requires_xformers": False,
            "attention_backend": "PyTorch scaled_dot_product_attention",
        },
    }
    if torch.cuda.is_available():
        report["cuda"].update(
            {
                "device_0": torch.cuda.get_device_name(0),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected jepa environment")
    if require_cuda and bool(config["extraction"]["use_bfloat16"]):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")
    report["dataset"] = inspect_dataset(config, decode_one=not skip_decode)
    if not skip_checkpoint:
        report["checkpoint"] = check_checkpoint(config)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    report = run_checks(
        config,
        require_cuda=args.require_cuda,
        skip_checkpoint=args.skip_checkpoint,
        skip_decode=args.skip_decode,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
