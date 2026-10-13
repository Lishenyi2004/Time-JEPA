from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

from grasp_eval_rjob.constants import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_TOTAL_PAIRS,
    EXPECTED_TOTAL_VIDEOS,
    FAMILIES,
)
from grasp_eval_rjob.dataset import discover_pairs


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    rate = str(stream.get("avg_frame_rate", "0/1"))
    fps = float(Fraction(rate)) if rate not in {"0/0", "N/A"} else float("nan")
    frames_raw = stream.get("nb_frames")
    if frames_raw in {None, "N/A"}:
        raise RuntimeError(f"ffprobe did not report nb_frames for {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "codec": str(stream.get("codec_name", "")),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "decoded_frames": int(frames_raw),
        "duration_seconds": float(stream.get("duration", 0.0)),
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-p{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def audit(
    dataset_root: str | Path,
    output_json: str | Path,
    *,
    workers: int = 16,
    archive: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_json).expanduser().resolve()
    manifest_path = output.with_name("data_manifest.csv")
    pairs = discover_pairs(root, require_full=True)

    video_rows: list[dict[str, Any]] = []
    for pair in pairs:
        for plausibility, relative_path in (
            ("possible", pair.possible_relative_path),
            ("impossible", pair.impossible_relative_path),
        ):
            video_rows.append(
                {
                    "dataset_index": pair.dataset_index,
                    "family": pair.family,
                    "pair_id": pair.pair_id,
                    "properties": "|".join(pair.properties),
                    "plausibility": plausibility,
                    "relative_path": relative_path,
                }
            )

    paths = [root / row["relative_path"] for row in video_rows]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        probes = list(executor.map(_probe_video, paths))
    for row, probe in zip(video_rows, probes):
        row.update(probe)

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in video_rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    for pair_id, rows in by_pair.items():
        if len(rows) != 2 or {row["plausibility"] for row in rows} != {
            "possible",
            "impossible",
        }:
            raise RuntimeError(f"Malformed pair {pair_id}: {rows}")
        first, second = rows
        fields = ("width", "height", "fps", "decoded_frames", "duration_seconds")
        mismatches = {
            field: (first[field], second[field])
            for field in fields
            if abs(float(first[field]) - float(second[field])) > 1e-6
        }
        if mismatches:
            raise RuntimeError(f"Paired video metadata mismatch for {pair_id}: {mismatches}")

    if len(pairs) != EXPECTED_TOTAL_PAIRS or len(video_rows) != EXPECTED_TOTAL_VIDEOS:
        raise RuntimeError(
            f"Unexpected corpus size: {len(pairs)} pairs, {len(video_rows)} videos"
        )
    family_counts = {
        family: sum(pair.family == family for pair in pairs) for family in FAMILIES
    }
    metadata_sets = {
        field: sorted({row[field] for row in video_rows})
        for field in (
            "codec",
            "width",
            "height",
            "fps",
            "decoded_frames",
            "duration_seconds",
        )
    }
    if not all(abs(float(fps) - 50.0) < 1e-6 for fps in metadata_sets["fps"]):
        raise RuntimeError(f"Expected 50 FPS GRASP videos, got {metadata_sets['fps']}")
    if min(int(value) for value in metadata_sets["decoded_frames"]) < 160:
        raise RuntimeError(
            "At least one video is too short for 16 frames sampled every 10 source frames"
        )

    columns = [
        "dataset_index",
        "family",
        "pair_id",
        "properties",
        "plausibility",
        "relative_path",
        "bytes",
        "sha256",
        "codec",
        "width",
        "height",
        "fps",
        "decoded_frames",
        "duration_seconds",
    ]
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(video_rows)
    manifest_text = buffer.getvalue()
    _atomic_text(manifest_path, manifest_text)
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    content_digest = hashlib.sha256()
    for row in sorted(video_rows, key=lambda value: value["relative_path"]):
        content_digest.update(str(row["relative_path"]).encode("utf-8"))
        content_digest.update(b"\0")
        content_digest.update(str(row["bytes"]).encode("ascii"))
        content_digest.update(b"\0")
        content_digest.update(str(row["sha256"]).encode("ascii"))
        content_digest.update(b"\n")

    archive_info = None
    if archive is not None:
        archive_path = Path(archive).expanduser().resolve()
        if archive_path.is_file():
            if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
                raise RuntimeError(
                    f"Official archive size mismatch: {archive_path.stat().st_size} "
                    f"!= {EXPECTED_ARCHIVE_BYTES}"
                )
            archive_info = {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
            }

    payload: dict[str, Any] = {
        "format_version": 1,
        "dataset": "GRASP Level 2 official release",
        "dataset_root": str(root),
        "pair_count": len(pairs),
        "video_count": len(video_rows),
        "family_pair_counts": family_counts,
        "metadata_unique_values": metadata_sets,
        "total_video_bytes": sum(int(row["bytes"]) for row in video_rows),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "content_fingerprint": content_digest.hexdigest(),
        "pairing_rule": (
            "same filename under P_<family> and IP_<family>; validated exact sets "
            "and paired stream metadata"
        ),
        "archive": archive_info,
    }
    _atomic_text(
        output,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--archive", default=None)
    args = parser.parse_args()
    payload = audit(
        args.dataset_root,
        args.output,
        workers=args.workers,
        archive=args.archive,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
