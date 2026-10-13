from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from grasp_eval_rjob.config import load_config, load_data_audit, protocol_digest
from grasp_eval_rjob.dataset import inspect_dataset
from intphys2_eval_rjob.model import build_empty_components, compare_state_shapes


PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "PyYAML": "yaml",
    "numpy": "numpy",
    "pandas": "pandas",
    "einops": "einops",
    "decord": "decord",
    "timm": "timm",
    "accelerate": "accelerate",
}


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_packages() -> dict[str, str]:
    versions: dict[str, str] = {}
    failures = []
    for distribution, module_name in PACKAGES.items():
        try:
            importlib.import_module(module_name)
            versions[distribution] = importlib.metadata.version(distribution)
        except Exception as error:
            failures.append(f"{distribution}/{module_name}: {error}")
    if failures:
        raise RuntimeError("Dependency check failed: " + "; ".join(failures))
    return versions


def check_checkpoint(
    config: dict[str, Any], *, verify_sha256: bool = False
) -> dict[str, Any]:
    path = Path(config["checkpoint"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    expected_sha = str(config["checkpoint_sha256"])
    actual_sha = _sha256_file(path) if verify_sha256 else None
    if actual_sha is not None and actual_sha != expected_sha:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {actual_sha} != {expected_sha}")

    frames_per_clip = int(config["evaluation"]["frames_per_clip"])
    encoder, target_encoder, predictor = build_empty_components(
        config["model"], frames_per_clip
    )
    modules = {
        "encoder": encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
    }
    with FakeTensorMode():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    counts = {}
    for section, module in modules.items():
        if section not in checkpoint:
            raise KeyError(f"Checkpoint {path} has no {section!r} section")
        report = compare_state_shapes(module, checkpoint[section])
        counts[section] = len(checkpoint[section])
        if any(report.values()):
            raise RuntimeError(
                f"{section} shape/key mismatch: "
                + json.dumps(report, ensure_ascii=False, default=str)[:8000]
            )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "expected_sha256": expected_sha,
        "verified_sha256": actual_sha,
        "epoch": checkpoint.get("epoch"),
        "tensor_counts": counts,
        "exact_match": True,
    }


def run_checks(
    config: dict[str, Any],
    *,
    require_cuda: bool,
    skip_checkpoint: bool,
    verify_checkpoint_sha256: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "packages": check_packages(),
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "torch_cuda": torch.version.cuda,
        },
        "protocol_digest": protocol_digest(config),
        "data_audit": load_data_audit(config),
    }
    if torch.cuda.is_available():
        report["cuda"].update(
            {
                "device_0": torch.cuda.get_device_name(0),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the selected jepa environment")
    if require_cuda and config["evaluation"]["use_bfloat16"]:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")

    report["dataset"] = inspect_dataset(config, decode_one=True)
    if not skip_checkpoint:
        report["checkpoint"] = check_checkpoint(
            config, verify_sha256=verify_checkpoint_sha256
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--verify-checkpoint-sha256", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    report = run_checks(
        config,
        require_cuda=args.require_cuda,
        skip_checkpoint=args.skip_checkpoint,
        verify_checkpoint_sha256=args.verify_checkpoint_sha256,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

