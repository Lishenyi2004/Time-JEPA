"""Attach timestamp and four-frame validity metadata without resampling RGB."""

from src.time_jepa.covla_law import load_law_series, tubelet_law_metadata
from src.time_jepa.covla_time_dataset import CoVLATimeJEPADataset
from functools import lru_cache

import numpy as np
from decord import VideoReader, cpu


@lru_cache(maxsize=256)
def load_video_frame_times(video_path):
    reader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    return np.asarray(reader.get_frame_timestamp(range(len(reader))), dtype=np.float64)[:, 0]


class CoVLALawDataset(CoVLATimeJEPADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.physical_misalignment_mode != "none":
            raise ValueError("Longitudinal law requires aligned state supervision")
        if self.physical_target_fields[:2] != ["vEgo", "aEgo"]:
            raise ValueError("Longitudinal law requires vEgo/aEgo as columns 0/1")
        if not self.return_physical_targets:
            raise ValueError("The law requires the existing anchored state head")

    def get_item_video(self, index):
        sample = super().get_item_video(index)
        if sample is None:
            return None
        buffer, label, clip_indices, physical = sample
        if any(physical["hflip"]):
            raise ValueError("The matched CoVLA law configuration disables horizontal flips")
        series = load_law_series(self.states_paths[index])
        video_frame_times = load_video_frame_times(self.samples[index])
        metadata = [tubelet_law_metadata(
            indices, self._get_num_video_frames(self.samples[index]), series,
            self.physical_tubelet_size, video_frame_times=video_frame_times,
        ) for indices in clip_indices]
        for key in ("law_dt", "law_valid"):
            physical[key] = [entry[key] for entry in metadata]
        return buffer, label, clip_indices, physical
