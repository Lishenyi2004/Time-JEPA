"""Run a bounded CoVLA integration check with externally supplied resources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quickstart/covla_probe_smoke.yaml"


def repository_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covla-root", help="External CoVLA dataset root.")
    parser.add_argument("--encoder-checkpoint", help="External compatible ViT-g/384 checkpoint.")
    parser.add_argument("--output-dir", help="New output directory, relative to the repository or absolute.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate the fitted quickstart probe without training.")
    parser.add_argument("--probe-checkpoint", help="Fitted probe; requires --eval-only.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Preview the bounded config and command; no resources or GPU required.")
    modes.add_argument("--check-only", action="store_true", help="Check resource paths and split metadata without loading weights or using a GPU.")
    args = parser.parse_args()
    if args.probe_checkpoint and not args.eval_only:
        parser.error("--probe-checkpoint requires --eval-only")
    config = yaml.safe_load(CONFIG.read_text())
    limits = {"epochs": 1, "batch_size": 1, "max_train_videos": 2, "max_val_videos": 1, "max_test_videos": 1}
    for key, value in limits.items():
        if config.get(key) != value:
            parser.error(f"Quickstart must retain {key}={value}")
    dataset = repository_path(args.covla_root or config["covla_root"])
    encoder = repository_path(args.encoder_checkpoint or config["encoder_checkpoint"])
    training_output = repository_path(args.output_dir or config["output_dir"])
    output = training_output / "eval" if args.eval_only else training_output
    probe = repository_path(args.probe_checkpoint) if args.probe_checkpoint else training_output / "best_probe.pt"
    command = [sys.executable, "-B", str(ROOT / "covla_probe_rjob/train_covla_probe.py"),
               "--config", str(CONFIG), "--covla-root", str(dataset),
               "--encoder-checkpoint", str(encoder), "--output-dir", str(output)]
    if args.eval_only:
        command.extend(["--eval-only", "--probe-checkpoint", str(probe)])
    print(json.dumps({"config": config, "dataset": str(dataset), "encoder_checkpoint": str(encoder),
                      "output_dir": str(output), "eval_only": args.eval_only,
                      "command": shlex.join(command)}, indent=2), flush=True)
    if args.dry_run:
        return 0
    errors = []
    if not dataset.is_dir():
        errors.append(f"Missing external CoVLA root: {dataset}")
    if not encoder.is_file():
        errors.append(f"Missing external encoder checkpoint: {encoder}")
    if args.eval_only and not probe.is_file():
        errors.append(f"Missing fitted probe: {probe}")
    split_dir = repository_path(config["split_dir"])
    for field in ("train_split", "val_split", "test_split", "covla_ood_test_split"):
        path = split_dir / f"covla_{config[field]}.json"
        try:
            rows = json.loads(path.read_text())
            if not isinstance(rows, list) or not rows:
                errors.append(f"Empty or invalid split: {path}")
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot read split {path}: {exc}. First run python scripts/prepare_metadata.py covla with authorized source data.")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Resource paths and split metadata exist. Video contents, checkpoint compatibility and CUDA are checked by the runner.", flush=True)
    if args.check_only:
        return 0
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output is not an empty directory; choose a new --output-dir to preserve earlier results.")
    subprocess.run(command, cwd=ROOT, check=True)
    metrics_path = output / "test_metrics.json"
    report = json.loads(metrics_path.read_text())
    if not isinstance(report.get("test"), dict) or not isinstance(report.get("ood_test"), dict):
        raise RuntimeError("Quickstart did not produce both ID and rainy OOD metrics")
    if not args.eval_only and not (output / "best_probe.pt").is_file():
        raise RuntimeError("Quickstart did not save a fitted probe")
    print(f"Quickstart completed. ID/OOD metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
