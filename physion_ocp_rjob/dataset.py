from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu

from app.vjepa_2_1.transforms import make_transforms


EXPECTED_RAW_SPLIT_SIZES = {"train": {5608}, "test": {1035, 1076}}
EXPECTED_EVAL_SPLIT_SIZES = {"train": 5608, "test": 1035}
SCENARIOS = ("collide", "drop", "support", "link", "roll", "contain", "dominoes")

# The released Physion evaluator removes these known-bad test videos before
# feature extraction.  Keep this list local so the protocol does not depend on
# importing the old imageio/ffmpeg data loader.
BUGGY_TEST_STIMULI = frozenset(
    """pilot-containment-cone-plate_0017
pilot-containment-cone-plate_0022
pilot-containment-cone-plate_0029
pilot-containment-cone-plate_0034
pilot-containment-multi-bowl_0042
pilot-containment-multi-bowl_0048
pilot-containment-vase_torus_0031
pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom_0005
pilot_it2_collision_non-sphere_box_0002
pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0004
pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0007
pilot_it2_drop_simple_box_0000
pilot_it2_drop_simple_box_0042
pilot_it2_drop_simple_tdw_1_dis_1_occ_0003
pilot_it2_rollingSliding_simple_collision_box_0008
pilot_it2_rollingSliding_simple_collision_box_large_force_0009
pilot_it2_rollingSliding_simple_collision_tdw_1_dis_1_occ_0002
pilot_it2_rollingSliding_simple_ledge_tdw_1_dis_1_occ_sphere_small_zone_0022
pilot_it2_rollingSliding_simple_ramp_box_small_zone_0006
pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0004
pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0017
pilot_linking_nl1-8_mg000_aCyl_bCyl_tdwroom1_long_a_0022
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom1_0012
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0006
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0010
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0029
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0036
pilot_linking_nl6_aNone_bCone_occ1_dis1_boxroom_0028
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0000
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0002
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0003
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0010
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0013
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0017
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0018
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0032
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0036
pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0021
pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0041
pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0006
pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0009""".splitlines()
)


def canonical_scenario(name: str) -> str:
    lowered = name.lower()
    if "rollingsliding" in lowered or "rolling_sliding" in lowered:
        return "roll"
    if "collision" in lowered:
        return "collide"
    if "towers" in lowered or "support" in lowered:
        return "support"
    if "drop" in lowered:
        return "drop"
    if "linking" in lowered or "link" in lowered:
        return "link"
    if "containment" in lowered or "contain" in lowered:
        return "contain"
    if "dominoes" in lowered:
        return "dominoes"
    raise ValueError(f"Cannot infer Physion scenario from stimulus name: {name}")


def ocp_frame_indices(
    stimulus_name: str,
    *,
    video_length: int,
    frame_gap_ms: int = 150,
    num_frames: int = 4,
) -> np.ndarray:
    """Reproduce the released Physion v1.5 OCP context sampling.

    Physion v1.5 MP4s are encoded at 100 fps, so one source frame is 10 ms.
    Fast collision clips expose frames through 150 ms; all other scenarios
    expose frames through 450 ms. Short contexts are left-padded with frame 0,
    matching the released evaluator.
    """
    if video_length <= 0:
        raise ValueError("video_length must be positive")
    if frame_gap_ms <= 0 or frame_gap_ms % 10:
        raise ValueError("frame_gap_ms must be a positive multiple of 10")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    lowered = stimulus_name.lower()
    fast_collision = "collision" in lowered and "roll" not in lowered
    max_frame = 15 if fast_collision else 45
    step = frame_gap_ms // 10
    indices = np.arange(max_frame, -1, -step, dtype=np.int64)
    indices = np.clip(indices, 0, video_length - 1)[::-1].copy()
    if len(indices) < num_frames:
        indices = np.concatenate(
            [np.full(num_frames - len(indices), indices[0], dtype=np.int64), indices]
        )
    elif len(indices) > num_frames:
        # The released 4x150 ms protocol lands exactly at num_frames. Keep the
        # most recent context if a custom denser setting asks for fewer frames.
        indices = indices[-num_frames:]
    return indices


def deterministic_temporal_shuffle(
    indices: np.ndarray, stimulus_name: str, seed: int
) -> np.ndarray:
    if len(indices) < 2:
        return indices.copy()
    digest = hashlib.sha256(f"{seed}:{stimulus_name}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(local_seed)
    permutation = rng.permutation(len(indices))
    shuffled = indices[permutation].copy()
    if np.array_equal(shuffled, indices):
        shuffled = np.roll(indices, 1).copy()
    return shuffled


def build_transform(resolution: int):
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.0, 1.0],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=int(resolution),
    )


@dataclass(frozen=True)
class PhysionRecord:
    global_index: int
    path: Path
    name: str
    label: int
    contact_frame: int
    scenario: str


def _stratified_cap(
    records: list[PhysionRecord], max_samples: int | None, seed: int
) -> list[PhysionRecord]:
    if max_samples is None or max_samples >= len(records):
        return records
    groups: dict[tuple[str, int], list[PhysionRecord]] = {}
    for record in records:
        groups.setdefault((record.scenario, record.label), []).append(record)
    rng = np.random.default_rng(int(seed))
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    selected: list[PhysionRecord] = []
    cursor = 0
    while len(selected) < max_samples:
        made_progress = False
        for key in keys:
            values = groups[key]
            if cursor < len(values):
                selected.append(values[cursor])
                made_progress = True
                if len(selected) == max_samples:
                    break
        if not made_progress:
            break
        cursor += 1
    return sorted(selected, key=lambda record: record.global_index)


class PhysionOCPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        temporal_order: str,
        resolution: int,
        frame_gap_ms: int = 150,
        num_frames: int = 4,
        seed: int = 239,
        max_samples: int | None = None,
        decord_num_threads: int = 1,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = str(split)
        self.temporal_order = str(temporal_order)
        self.frame_gap_ms = int(frame_gap_ms)
        self.num_frames = int(num_frames)
        self.seed = int(seed)
        self.decord_num_threads = int(decord_num_threads)
        self.transform = build_transform(int(resolution))
        if self.split not in {"train", "test"}:
            raise ValueError(f"split must be train or test, got {self.split!r}")
        if self.temporal_order not in {"ordered", "shuffled"}:
            raise ValueError(
                f"temporal_order must be ordered or shuffled, got {self.temporal_order!r}"
            )

        split_root = self.dataset_root / "physion_mp4s" / self.split
        labels_path = split_root / "all_video_labels.json"
        if not split_root.is_dir() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Missing Physion v1.5 split or labels under {split_root}"
            )
        with labels_path.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)
        all_paths = sorted(split_root.glob("*.mp4"))
        if len(all_paths) not in EXPECTED_RAW_SPLIT_SIZES[self.split]:
            raise RuntimeError(
                f"Physion {self.split} has {len(all_paths)} raw MP4s, expected one "
                f"of {sorted(EXPECTED_RAW_SPLIT_SIZES[self.split])}"
            )
        path_names = {path.stem for path in all_paths}
        unexpected_labels = sorted(set(labels) - path_names)
        missing_labels = sorted(path_names - set(labels))
        if unexpected_labels or missing_labels:
            raise RuntimeError(
                "Physion labels/MP4s disagree: "
                f"missing_labels={missing_labels[:8]}, "
                f"missing_mp4s={unexpected_labels[:8]}"
            )
        paths = [
            path
            for path in all_paths
            if self.split != "test" or path.stem not in BUGGY_TEST_STIMULI
        ]
        records: list[PhysionRecord] = []
        for global_index, path in enumerate(paths):
            if path.stem not in labels:
                raise KeyError(f"No OCP label for {path.name} in {labels_path}")
            item = labels[path.stem]
            records.append(
                PhysionRecord(
                    global_index=global_index,
                    path=path,
                    name=path.stem,
                    label=int(item["label"]),
                    contact_frame=int(item["contact_frame_label"]),
                    scenario=canonical_scenario(path.stem),
                )
            )
        self.full_size = len(records)
        self.raw_size = len(all_paths)
        if self.full_size != EXPECTED_EVAL_SPLIT_SIZES[self.split]:
            raise RuntimeError(
                f"Physion {self.split} has {self.full_size} protocol videos, expected "
                f"{EXPECTED_EVAL_SPLIT_SIZES[self.split]}"
            )
        self.records = _stratified_cap(records, max_samples, self.seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        reader = VideoReader(
            str(record.path),
            ctx=cpu(0),
            num_threads=max(1, self.decord_num_threads),
        )
        frame_indices = ocp_frame_indices(
            record.name,
            video_length=len(reader),
            frame_gap_ms=self.frame_gap_ms,
            num_frames=self.num_frames,
        )
        if self.temporal_order == "shuffled":
            frame_indices = deterministic_temporal_shuffle(
                frame_indices, record.name, self.seed
            )
        frames = reader.get_batch(frame_indices).asnumpy()
        clip = self.transform(torch.from_numpy(frames))
        return {
            "clip": clip,
            "global_index": record.global_index,
            "name": record.name,
            "relative_path": str(record.path.relative_to(self.dataset_root)),
            "label": record.label,
            "contact_frame": record.contact_frame,
            "scenario": record.scenario,
            "frame_indices": torch.from_numpy(frame_indices.copy()),
        }


def inspect_dataset(config: dict[str, Any], decode_one: bool = True) -> dict[str, Any]:
    extraction = config["extraction"]
    report: dict[str, Any] = {"tasks": []}
    for task in extraction["tasks"]:
        split = task["split"]
        max_samples = extraction.get(
            "max_train_samples" if split == "train" else "max_test_samples"
        )
        dataset = PhysionOCPDataset(
            dataset_root=config["dataset_root"],
            split=split,
            temporal_order=task["temporal_order"],
            resolution=int(config["model"]["resolution"]),
            frame_gap_ms=int(extraction["frame_gap_ms"]),
            num_frames=int(extraction["num_frames"]),
            seed=int(extraction["seed"]),
            max_samples=max_samples,
            decord_num_threads=int(extraction["decord_num_threads"]),
        )
        task_report: dict[str, Any] = {
            "split": split,
            "temporal_order": task["temporal_order"],
            "videos": len(dataset),
            "raw_split_videos": dataset.raw_size,
            "full_split_videos": dataset.full_size,
            "excluded_buggy_videos": dataset.raw_size - dataset.full_size,
            "expected_full_split_videos": EXPECTED_EVAL_SPLIT_SIZES[split],
        }
        if dataset.full_size != EXPECTED_EVAL_SPLIT_SIZES[split]:
            raise RuntimeError(
                f"Physion {split} has {dataset.full_size} MP4s, expected "
                f"{EXPECTED_EVAL_SPLIT_SIZES[split]}"
            )
        if decode_one and len(dataset):
            sample = dataset[0]
            task_report["sample"] = {
                key: list(value.shape) if key == "clip" else value.tolist()
                if key == "frame_indices"
                else value
                for key, value in sample.items()
                if key not in {"relative_path"}
            }
        report["tasks"].append(task_report)
    return report
