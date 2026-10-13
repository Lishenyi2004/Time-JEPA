from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SHARD_RE = re.compile(r"rank_(\d+)_of_(\d+)\.h5$")


def task_directory(output_dir: str | Path, split: str, temporal_order: str) -> Path:
    return Path(output_dir).expanduser().resolve() / "features" / f"{split}_{temporal_order}"


def shard_path(
    output_dir: str | Path,
    split: str,
    temporal_order: str,
    rank: int,
    world_size: int,
) -> Path:
    return task_directory(output_dir, split, temporal_order) / (
        f"rank_{rank:05d}_of_{world_size:05d}.h5"
    )


def check_stale_shards(directory: Path, world_size: int) -> None:
    stale: list[str] = []
    for path in directory.glob("rank_*_of_*.h5"):
        match = SHARD_RE.fullmatch(path.name)
        if match and int(match.group(2)) != int(world_size):
            stale.append(path.name)
    if stale:
        raise RuntimeError(
            "Feature directory contains shards from another world size; move them "
            f"aside before this run: {stale[:12]}"
        )


def matching_complete_shard(
    path: Path,
    *,
    digest: str,
    split: str,
    temporal_order: str,
    rank: int,
    world_size: int,
    dataset_size: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            attrs = handle.attrs
            return bool(attrs.get("complete", 0)) and all(
                (
                    str(attrs.get("protocol_digest", "")) == str(digest),
                    str(attrs.get("split", "")) == str(split),
                    str(attrs.get("temporal_order", "")) == str(temporal_order),
                    int(attrs.get("rank", -1)) == int(rank),
                    int(attrs.get("world_size", -1)) == int(world_size),
                    int(attrs.get("dataset_size", -1)) == int(dataset_size),
                )
            )
    except OSError:
        return False


class H5FeatureShardWriter:
    """Stream one rank's features to an atomic, self-describing HDF5 shard."""

    def __init__(
        self,
        destination: Path,
        *,
        row_count: int,
        feature_dim: int,
        num_frames: int,
        feature_dtype: str,
        attributes: dict[str, Any],
    ) -> None:
        self.destination = destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = destination.with_name(
            f".{destination.name}.tmp-r{attributes.get('rank', 0)}-p{os.getpid()}"
        )
        self.row_count = int(row_count)
        self.feature_dim = int(feature_dim)
        self.position = 0
        self._finished = False
        self.handle = h5py.File(self.temporary, "w")
        itemsize = np.dtype(feature_dtype).itemsize
        target_chunk_bytes = 16 * 1024 * 1024
        rows_for_target = max(1, target_chunk_bytes // (self.feature_dim * itemsize))
        chunk_rows = min(64, rows_for_target, self.row_count) if self.row_count else None
        feature_chunks = (
            (chunk_rows, self.feature_dim) if chunk_rows is not None else None
        )
        self.features = self.handle.create_dataset(
            "features",
            shape=(self.row_count, self.feature_dim),
            dtype=np.dtype(feature_dtype),
            chunks=feature_chunks,
        )
        self.global_index = self.handle.create_dataset(
            "global_index", shape=(self.row_count,), dtype=np.int64
        )
        self.label = self.handle.create_dataset(
            "label", shape=(self.row_count,), dtype=np.int8
        )
        self.contact_frame = self.handle.create_dataset(
            "contact_frame", shape=(self.row_count,), dtype=np.int32
        )
        self.frame_indices = self.handle.create_dataset(
            "frame_indices", shape=(self.row_count, int(num_frames)), dtype=np.int32
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self.name = self.handle.create_dataset(
            "name", shape=(self.row_count,), dtype=string_dtype
        )
        self.scenario = self.handle.create_dataset(
            "scenario", shape=(self.row_count,), dtype=string_dtype
        )
        self.relative_path = self.handle.create_dataset(
            "relative_path", shape=(self.row_count,), dtype=string_dtype
        )
        for key, value in attributes.items():
            if value is not None:
                self.handle.attrs[key] = value
        self.handle.attrs["format_version"] = 1
        self.handle.attrs["feature_dim"] = self.feature_dim
        self.handle.attrs["feature_dtype"] = str(feature_dtype)
        self.handle.attrs["shard_rows"] = self.row_count
        self.handle.attrs["complete"] = 0

    def write(self, features: np.ndarray, batch: dict[str, Any]) -> None:
        array = np.asarray(features)
        if array.ndim != 2 or array.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature batch [B,{self.feature_dim}], got {array.shape}"
            )
        batch_size = int(array.shape[0])
        end = self.position + batch_size
        if end > self.row_count:
            raise RuntimeError("Attempted to write beyond the allocated feature shard")
        rows = slice(self.position, end)
        self.features[rows] = array
        self.global_index[rows] = np.asarray(batch["global_index"], dtype=np.int64)
        self.label[rows] = np.asarray(batch["label"], dtype=np.int8)
        self.contact_frame[rows] = np.asarray(batch["contact_frame"], dtype=np.int32)
        self.frame_indices[rows] = np.asarray(batch["frame_indices"], dtype=np.int32)
        self.name[rows] = np.asarray(list(batch["name"]), dtype=object)
        self.scenario[rows] = np.asarray(list(batch["scenario"]), dtype=object)
        self.relative_path[rows] = np.asarray(list(batch["relative_path"]), dtype=object)
        self.position = end

    def finish(self) -> Path:
        if self.position != self.row_count:
            raise RuntimeError(
                f"Incomplete feature shard: wrote {self.position}/{self.row_count} rows"
            )
        self.handle.attrs["complete"] = 1
        self.handle.flush()
        self.handle.close()
        os.replace(self.temporary, self.destination)
        self._finished = True
        return self.destination

    def close_failed(self) -> None:
        if self._finished:
            return
        try:
            self.handle.close()
        finally:
            self.temporary.unlink(missing_ok=True)


def atomic_json_dump(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-p{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, destination)


def _decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


@dataclass
class FeatureSet:
    features: np.ndarray
    global_index: np.ndarray
    label: np.ndarray
    contact_frame: np.ndarray
    frame_indices: np.ndarray
    name: np.ndarray
    scenario: np.ndarray
    relative_path: np.ndarray
    split: str
    temporal_order: str
    protocol_digest: str

    def subset(self, indices: np.ndarray) -> "FeatureSet":
        indices = np.asarray(indices, dtype=np.int64)
        return FeatureSet(
            features=self.features[indices],
            global_index=self.global_index[indices],
            label=self.label[indices],
            contact_frame=self.contact_frame[indices],
            frame_indices=self.frame_indices[indices],
            name=self.name[indices],
            scenario=self.scenario[indices],
            relative_path=self.relative_path[indices],
            split=self.split,
            temporal_order=self.temporal_order,
            protocol_digest=self.protocol_digest,
        )


def discover_complete_shards(
    output_dir: str | Path,
    split: str,
    temporal_order: str,
    expected_digest: str | None = None,
) -> list[Path]:
    directory = task_directory(output_dir, split, temporal_order)
    candidates: list[tuple[int, int, Path]] = []
    for path in directory.glob("rank_*_of_*.h5"):
        match = SHARD_RE.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), int(match.group(2)), path))
    if not candidates:
        raise FileNotFoundError(
            f"No feature shards found under {directory}. The probe only consumes "
            "cached encoder features; run feature extraction for this same config "
            "first (RUN_EXTRACTION=1)."
        )
    world_sizes = {world for _, world, _ in candidates}
    if len(world_sizes) != 1:
        raise RuntimeError(f"Mixed feature-shard world sizes under {directory}: {world_sizes}")
    world_size = world_sizes.pop()
    ranks = [rank for rank, _, _ in candidates]
    if sorted(ranks) != list(range(world_size)):
        raise RuntimeError(
            f"Incomplete feature shards under {directory}: found ranks {sorted(ranks)}, "
            f"expected 0..{world_size - 1}"
        )
    paths = [path for _, _, path in sorted(candidates)]
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", 0)):
                raise RuntimeError(f"Incomplete feature shard: {path}")
            digest = str(handle.attrs.get("protocol_digest", ""))
            if expected_digest is not None and digest != expected_digest:
                raise RuntimeError(
                    f"Feature shard {path} has protocol {digest}, expected {expected_digest}"
                )
    return paths


def load_feature_set(
    output_dir: str | Path,
    split: str,
    temporal_order: str,
    *,
    expected_digest: str | None = None,
    read_chunk_rows: int = 64,
) -> FeatureSet:
    paths = discover_complete_shards(
        output_dir, split, temporal_order, expected_digest=expected_digest
    )
    metadata: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "global_index",
            "label",
            "contact_frame",
            "frame_indices",
            "name",
            "scenario",
            "relative_path",
        )
    }
    feature_dim: int | None = None
    feature_dtype: np.dtype[Any] | None = None
    digest: str | None = None
    dataset_sizes: set[int] = set()
    for path in paths:
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("split")) != split:
                raise RuntimeError(f"Wrong split metadata in {path}")
            if str(handle.attrs.get("temporal_order")) != temporal_order:
                raise RuntimeError(f"Wrong temporal-order metadata in {path}")
            current_dim = int(handle.attrs["feature_dim"])
            current_dtype = np.dtype(handle.attrs["feature_dtype"])
            feature_dim = current_dim if feature_dim is None else feature_dim
            feature_dtype = current_dtype if feature_dtype is None else feature_dtype
            if feature_dim != current_dim or feature_dtype != current_dtype:
                raise RuntimeError(f"Inconsistent feature shape/dtype in {path}")
            current_digest = str(handle.attrs["protocol_digest"])
            digest = current_digest if digest is None else digest
            if current_digest != digest:
                raise RuntimeError(f"Mixed protocol digests in {path.parent}")
            dataset_sizes.add(int(handle.attrs["dataset_size"]))
            for key in metadata:
                values = handle[key][...]
                if key in {"name", "scenario", "relative_path"}:
                    values = _decode_strings(values)
                metadata[key].append(values)
    if len(dataset_sizes) != 1:
        raise RuntimeError(f"Inconsistent dataset_size attributes: {dataset_sizes}")
    expected_rows = dataset_sizes.pop()
    combined = {key: np.concatenate(values, axis=0) for key, values in metadata.items()}
    global_index = combined["global_index"].astype(np.int64, copy=False)
    if len(global_index) != expected_rows:
        raise RuntimeError(
            f"Feature shards contain {len(global_index)} rows, expected {expected_rows}"
        )
    if len(np.unique(global_index)) != len(global_index):
        raise RuntimeError("Duplicate global_index values across feature shards")
    ordering = np.argsort(global_index)
    sorted_global = global_index[ordering]
    assert feature_dim is not None and feature_dtype is not None and digest is not None
    features = np.empty((expected_rows, feature_dim), dtype=feature_dtype)
    for path in paths:
        with h5py.File(path, "r") as handle:
            shard_indices = handle["global_index"][...].astype(np.int64, copy=False)
            positions = np.searchsorted(sorted_global, shard_indices)
            source = handle["features"]
            for start in range(0, len(shard_indices), max(1, int(read_chunk_rows))):
                end = min(start + max(1, int(read_chunk_rows)), len(shard_indices))
                features[positions[start:end]] = source[start:end]
    return FeatureSet(
        features=features,
        global_index=sorted_global,
        label=combined["label"][ordering].astype(np.int64, copy=False),
        contact_frame=combined["contact_frame"][ordering].astype(np.int64, copy=False),
        frame_indices=combined["frame_indices"][ordering].astype(np.int64, copy=False),
        name=combined["name"][ordering],
        scenario=combined["scenario"][ordering],
        relative_path=combined["relative_path"][ordering],
        split=split,
        temporal_order=temporal_order,
        protocol_digest=digest,
    )
