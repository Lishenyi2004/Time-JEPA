from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu

from app.vjepa_2_1.transforms import make_transforms
from grasp_eval_rjob.constants import (
    EXPECTED_PAIRS_PER_FAMILY,
    EXPECTED_TOTAL_PAIRS,
    FAMILIES,
    FAMILY_TO_PROPERTIES,
)


@dataclass(frozen=True)
class PairDescriptor:
    dataset_index: int
    family: str
    pair_id: str
    file_name: str
    possible_relative_path: str
    impossible_relative_path: str
    properties: tuple[str, ...]


def _video_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing GRASP condition directory: {directory}")
    paths = [path for path in directory.iterdir() if path.is_file()]
    non_mp4 = sorted(path.name for path in paths if path.suffix.lower() != ".mp4")
    if non_mp4:
        raise RuntimeError(f"Unexpected non-MP4 files in {directory}: {non_mp4[:12]}")
    by_name = {path.name: path for path in paths}
    if len(by_name) != len(paths):
        raise RuntimeError(f"Duplicate filenames in {directory}")
    return by_name


def _natural_video_key(name: str) -> tuple[int, int | str, str]:
    stem = Path(name).stem
    if stem.isdigit():
        return (0, int(stem), name)
    return (1, stem, name)


def discover_pairs(
    dataset_root: str | Path,
    *,
    require_full: bool = True,
) -> list[PairDescriptor]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing GRASP Level-2 root: {root}")

    expected_directories = {
        f"{prefix}_{family}" for family in FAMILIES for prefix in ("P", "IP")
    }
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    missing_directories = sorted(expected_directories - actual_directories)
    extra_directories = sorted(actual_directories - expected_directories)
    if missing_directories or extra_directories:
        raise RuntimeError(
            "GRASP Level-2 directory mismatch: "
            f"missing={missing_directories}, extra={extra_directories}"
        )

    pairs: list[PairDescriptor] = []
    for family in FAMILIES:
        possible = _video_files(root / f"P_{family}")
        impossible = _video_files(root / f"IP_{family}")
        possible_names = set(possible)
        impossible_names = set(impossible)
        if possible_names != impossible_names:
            raise RuntimeError(
                f"Pair filenames differ for {family}: "
                f"possible_only={sorted(possible_names - impossible_names)[:12]}, "
                f"impossible_only={sorted(impossible_names - possible_names)[:12]}"
            )
        if require_full and len(possible_names) != EXPECTED_PAIRS_PER_FAMILY:
            raise RuntimeError(
                f"Expected {EXPECTED_PAIRS_PER_FAMILY} pairs for {family}, "
                f"found {len(possible_names)}"
            )
        for file_name in sorted(possible_names, key=_natural_video_key):
            pairs.append(
                PairDescriptor(
                    dataset_index=len(pairs),
                    family=family,
                    pair_id=f"{family}:{Path(file_name).stem}",
                    file_name=file_name,
                    possible_relative_path=str(
                        possible[file_name].relative_to(root)
                    ),
                    impossible_relative_path=str(
                        impossible[file_name].relative_to(root)
                    ),
                    properties=FAMILY_TO_PROPERTIES[family],
                )
            )
    if require_full and len(pairs) != EXPECTED_TOTAL_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_PAIRS} total pairs, found {len(pairs)}"
        )
    return pairs


def build_transform(resolution: int):
    # Exact deterministic spatial protocol used by the released JEPA GRASP
    # evaluator: full-scale square crop plus ImageNet normalization.
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.0, 1.0],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=int(resolution),
    )


class GRASPPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        frame_step: int,
        resolution: int,
        frames_per_clip: int,
        max_pairs: int | None = None,
        decord_num_threads: int = 1,
        require_full: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.frame_step = int(frame_step)
        self.frames_per_clip = int(frames_per_clip)
        self.decord_num_threads = int(decord_num_threads)
        self.transform = build_transform(int(resolution))
        self.pairs = discover_pairs(self.dataset_root, require_full=require_full)
        if max_pairs is not None:
            self.pairs = self.pairs[: int(max_pairs)]

    def __len__(self) -> int:
        return len(self.pairs)

    def _decode(self, relative_path: str) -> tuple[torch.Tensor, int, int, float]:
        path = self.dataset_root / relative_path
        reader = VideoReader(
            str(path),
            ctx=cpu(0),
            num_threads=max(1, self.decord_num_threads),
        )
        decoded_frames = len(reader)
        frame_indices = np.arange(0, decoded_frames, self.frame_step, dtype=np.int64)
        if len(frame_indices) < self.frames_per_clip:
            raise RuntimeError(
                f"{path} has only {len(frame_indices)} sampled frames; "
                f"need {self.frames_per_clip}"
            )
        frames = reader.get_batch(frame_indices).asnumpy()
        clip = self.transform(torch.from_numpy(frames))
        return clip, decoded_frames, len(frame_indices), float(reader.get_avg_fps())

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        possible = self._decode(pair.possible_relative_path)
        impossible = self._decode(pair.impossible_relative_path)
        if possible[0].shape != impossible[0].shape:
            raise RuntimeError(
                f"Paired transformed clips differ for {pair.pair_id}: "
                f"{tuple(possible[0].shape)} vs {tuple(impossible[0].shape)}"
            )
        return {
            "clips": torch.stack((possible[0], impossible[0]), dim=0),
            "dataset_index": pair.dataset_index,
            "family": pair.family,
            "pair_id": pair.pair_id,
            "file_name": pair.file_name,
            "possible_relative_path": pair.possible_relative_path,
            "impossible_relative_path": pair.impossible_relative_path,
            "properties": list(pair.properties),
            "decoded_frames": torch.tensor(
                [possible[1], impossible[1]], dtype=torch.int32
            ),
            "sampled_frames": torch.tensor(
                [possible[2], impossible[2]], dtype=torch.int32
            ),
            "fps": torch.tensor([possible[3], impossible[3]], dtype=torch.float32),
        }


def inspect_dataset(config: dict[str, Any], decode_one: bool = True) -> dict[str, Any]:
    evaluation = config["evaluation"]
    dataset = GRASPPairDataset(
        dataset_root=config["dataset_root"],
        frame_step=int(evaluation["frame_step"]),
        resolution=int(config["model"]["resolution"]),
        frames_per_clip=int(evaluation["frames_per_clip"]),
        max_pairs=evaluation.get("max_pairs"),
        decord_num_threads=int(evaluation["decord_num_threads"]),
    )
    report: dict[str, Any] = {
        "pairs": len(dataset),
        "full_pairs": EXPECTED_TOTAL_PAIRS,
        "root": str(dataset.dataset_root),
    }
    if decode_one and len(dataset):
        sample = dataset[0]
        report["sample"] = {
            "pair_id": sample["pair_id"],
            "clips_shape": list(sample["clips"].shape),
            "decoded_frames": sample["decoded_frames"].tolist(),
            "sampled_frames": sample["sampled_frames"].tolist(),
            "fps": sample["fps"].tolist(),
        }
    return report


def descriptor_dict(pair: PairDescriptor) -> dict[str, Any]:
    payload = asdict(pair)
    payload["properties"] = list(pair.properties)
    return payload

