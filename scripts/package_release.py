"""Build a clean directory and deterministic archive from the release manifest."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".pth.tar", ".ckpt", ".safetensors", ".mp4", ".avi", ".parquet", ".h5", ".hdf5", ".zip", ".urdf")
FORBIDDEN_DIRS = {".git", "__pycache__", ".pytest_cache", "data", "outputs", "logs", "wandb", "releases", "private_metadata"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Time-JEPA-0.1.1-preprint")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "releases")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
        parser.error("--name must be a simple directory name")
    manifest_path = ROOT / "release-manifest.json"
    records = json.loads(manifest_path.read_text())
    policy = json.loads((ROOT / "metadata-policy.json").read_text())
    local_only = {item["path"] for item in policy["restricted_metadata"]}
    local_prefixes = tuple(policy["local_only_prefixes"])
    files = {}
    for record in records:
        relative = record["path"]
        if relative in local_only or relative.startswith(local_prefixes):
            raise ValueError(f"Local-only metadata in manifest: {relative}")
        parts = PurePosixPath(relative).parts
        if not parts or PurePosixPath(relative).is_absolute() or ".." in parts or "\\" in relative:
            raise ValueError(f"Unsafe manifest path: {relative}")
        if any(p in FORBIDDEN_DIRS or p.startswith("outputs_") for p in parts) or relative.endswith(FORBIDDEN_SUFFIXES):
            raise ValueError(f"Excluded resource in manifest: {relative}")
        source = ROOT / relative
        if any(p.is_symlink() for p in [source, *source.parents] if p == ROOT or ROOT in p.parents):
            raise ValueError(f"Symlink in published path: {relative}")
        if not source.is_file():
            raise ValueError(f"Missing published file: {relative}")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["release_sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}")
        if relative in files:
            raise ValueError(f"Duplicate manifest path: {relative}")
        files[relative] = data
    files["release-manifest.json"] = manifest_path.read_bytes()
    destination = args.output_dir.resolve()
    folder = destination / args.name
    archive = destination / f"{args.name}.tar.gz"
    checksum = destination / f"{args.name}.tar.gz.sha256"
    if any(p.exists() or p.is_symlink() for p in (folder, archive, checksum)):
        parser.error("Release destination already exists; choose a new --name or --output-dir.")
    destination.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    for relative, data in sorted(files.items()):
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o644)
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for relative, data in sorted(files.items()):
                    info = tarfile.TarInfo(f"{args.name}/{relative}")
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(f"{digest}  {archive.name}\n")
    print(json.dumps({"directory": str(folder), "archive": str(archive), "checksum": str(checksum),
                      "sha256": digest, "files": len(files), "archive_bytes": archive.stat().st_size,
                      "weights_and_raw_data_included": False, "symlinks_included": False}, indent=2))


if __name__ == "__main__":
    main()
