"""Generate CoVLA/Ego-Exo4D splits locally from authorized source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def verify(recipe, output):
    errors = []
    for expected in recipe["expected_splits"]:
        path = output / expected["filename"]
        try:
            raw = path.read_bytes()
            values = json.loads(raw)
            if not isinstance(values, list) or len(values) != expected["count"]:
                errors.append(f"Split count mismatch: {path}")
            if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                errors.append(f"Split checksum mismatch: {path}")
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot verify {path}: {exc}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("Use the same complete source snapshot and preprocessing as the paper; do not treat a mismatched split as the paper split.", file=sys.stderr)
        return 1
    print(f"Verified {len(recipe['expected_splits'])} split files against the fixed paper protocol.")
    return 0


def main():
    recipes = json.loads((ROOT / "metadata-protocol.json").read_text())["domains"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=sorted(recipes))
    parser.add_argument("--dataset-root", help="Existing source dataset downloaded under its official agreement.")
    parser.add_argument("--output-dir", help="Destination for locally generated splits and manifest.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Preview without accessing data or creating files.")
    modes.add_argument("--verify-only", action="store_true", help="Check existing split counts and hashes without generating files.")
    args = parser.parse_args()
    recipe = recipes[args.domain]
    dataset = local_path(args.dataset_root or recipe["dataset_root"])
    output = local_path(args.output_dir or recipe["output_dir"])
    command = [sys.executable, "-B", str(ROOT / recipe["generator"]), recipe["root_flag"],
               str(dataset), "--output-dir", str(output), *recipe["arguments"]]
    if args.dry_run:
        print(json.dumps({"domain": args.domain, "command": shlex.join(command),
                          "expected_splits": recipe["expected_splits"],
                          "downloads_data": False, "uses_gpu": False}, indent=2))
        return 0
    if args.verify_only:
        return verify(recipe, output)
    if not dataset.is_dir():
        parser.error(f"Missing authorized source dataset: {dataset}")
    if any((output / item["filename"]).exists() for item in recipe["expected_splits"]):
        parser.error("Split files already exist; use --verify-only or a new --output-dir. Existing splits are preserved.")
    manifest_name = "covla_split_manifest_weather_ood.json" if args.domain == "covla" else "egoexo_exo124_view3ood_split_manifest.json"
    if (output / manifest_name).exists():
        parser.error("Split manifest already exists; choose a new --output-dir.")
    subprocess.run(command, cwd=ROOT, check=True)
    return verify(recipe, output)


if __name__ == "__main__":
    raise SystemExit(main())
