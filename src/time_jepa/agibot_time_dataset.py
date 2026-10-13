import hashlib
import json
import os
import pathlib
import subprocess
import warnings
from functools import lru_cache
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu

from src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.time_jepa.agibot_derived_targets import (
    SERIES_DERIVED_MIN_STATE_LEN,
    WHOLEBODY_CARTESIAN_V1_FIELDS,
    WHOLEBODY_CARTESIAN_V2_FIELDS,
    derive_agibot_series_field,
    is_agibot_series_derived_field,
)
from src.time_jepa.time_sampling import (
    build_sample_times,
    infer_video_fps,
    max_start_time,
    nearest_time_indices,
    times_to_frame_indices,
)


logger = getLogger()


TARGET_PRESETS = {
    "joint_effector_v1": [
        "state/joint/position[0]",
        "state/joint/position[1]",
        "state/joint/position[2]",
        "state/joint/position[3]",
        "state/joint/velocity[0]",
        "state/joint/velocity[1]",
        "state/joint/velocity[2]",
        "state/joint/velocity[3]",
        "state/end/velocity_norm",
        "state/left_effector/position",
        "state/right_effector/position",
    ],
    "joint_effector_lowdim_v1": [
        "state/joint/velocity[0]",
        "state/joint/velocity[1]",
        "state/joint/velocity[2]",
        "state/end/velocity_norm",
        "state/left_effector/position",
        "state/right_effector/position",
    ],
    "bimanual_compact_v1": [
        "state/joint/velocity_norm_left",
        "state/joint/velocity_norm_right",
        "state/end/velocity_linear_norm_left",
        "state/end/velocity_linear_norm_right",
        "state/end/arm_position[1]",
        "state/end/arm_position[4]",
        "state/left_effector/position",
        "state/right_effector/position",
    ],
    # Left-arm-only supervision (right arm held out as unseen shot target).
    # Kinematics + gripper + internal force (effort) + external contact force
    # (wrench), all for the LEFT arm only. The right-arm dual lives in the
    # probe-side "left_arm_unseen_right_v1" shot group.
    "left_arm_seen_v1": [
        "state/joint/velocity_norm_left",
        "state/end/velocity_linear_norm_left",
        "state/end/arm_position[1]",
        "state/left_effector/position",
        "state/joint/effort_norm_left",
        "state/end/wrench_force_norm_left",
    ],
    # Versioned no-wrench ablation. Keep the v1 preset untouched so its encoder
    # checkpoints and 6-D results remain exactly reproducible.
    "left_arm_seen_v2": [
        "state/joint/velocity_norm_left",
        "state/end/velocity_linear_norm_left",
        "state/end/arm_position[1]",
        "state/left_effector/position",
        "state/joint/effort_norm_left",
    ],
    # Versioned 14-D panel spanning base motion, torso configuration, both
    # end-effector poses/velocities, and both grippers.
    "wholebody_cartesian_v1": list(WHOLEBODY_CARTESIAN_V1_FIELDS),
    # Versioned 18-D panel. Compared with v1, the two unsigned endpoint-speed
    # norms are replaced by six signed Cartesian endpoint velocities so the
    # readout can participate directly in arm reach-rate physics.
    "wholebody_cartesian_v2": list(WHOLEBODY_CARTESIAN_V2_FIELDS),
}


STATE_SLICES = {
    "state/left_effector/position": (0, 1),
    "state/right_effector/position": (1, 2),
    "state/end/wrench": (2, 14),
    "state/end/velocity": (14, 26),
    "state/end/arm_position": (34, 40),
    "state/joint/position": (40, 54),
    "state/joint/effort": (54, 68),
    "state/joint/velocity": (68, 82),
}

AGIBOT_DERIVED_TARGETS = {
    "state/joint/position_norm_left": ("state/joint/position", 0, 7),
    "state/joint/position_norm_right": ("state/joint/position", 7, 14),
    "state/joint/velocity_norm_left": ("state/joint/velocity", 0, 7),
    "state/joint/velocity_norm_right": ("state/joint/velocity", 7, 14),
    "state/end/velocity_linear_norm_left": ("state/end/velocity", 0, 3),
    "state/end/velocity_linear_norm_right": ("state/end/velocity", 6, 9),
    "state/joint/effort_norm_left": ("state/joint/effort", 0, 7),
    "state/joint/effort_norm_right": ("state/joint/effort", 7, 14),
    "state/end/wrench_force_norm_left": ("state/end/wrench", 0, 3),
    "state/end/wrench_force_norm_right": ("state/end/wrench", 6, 9),
}

# Some AgiBot shards expose 162-D state vectors while others expose 169-D
# vectors. The current supervised slices only need the first 82 dims.
AGIBOT_STATE_MIN_LEN = max(
    STATE_SLICES["state/end/wrench"][1],
    STATE_SLICES["state/end/velocity"][1],
    STATE_SLICES["state/end/arm_position"][1],
    STATE_SLICES["state/joint/position"][1],
    STATE_SLICES["state/joint/effort"][1],
    STATE_SLICES["state/joint/velocity"][1],
)

AGIBOT_TARGET_ALIASES = {
    "joint_position_norm_left": "state/joint/position_norm_left",
    "joint_position_norm_right": "state/joint/position_norm_right",
    "joint_velocity_norm_left": "state/joint/velocity_norm_left",
    "joint_velocity_norm_right": "state/joint/velocity_norm_right",
    "end_velocity_linear_norm_left": "state/end/velocity_linear_norm_left",
    "end_velocity_linear_norm_right": "state/end/velocity_linear_norm_right",
    "arm_position_left": "state/end/arm_position[1]",
    "arm_position_right": "state/end/arm_position[4]",
    "joint_effort_norm_left": "state/joint/effort_norm_left",
    "joint_effort_norm_right": "state/joint/effort_norm_right",
    "end_wrench_force_norm_left": "state/end/wrench_force_norm_left",
    "end_wrench_force_norm_right": "state/end/wrench_force_norm_right",
    "base_translation_speed": "state/robot/translation_speed",
    "base_angular_speed": "state/robot/angular_speed",
    "waist_pc1": "state/waist/position_pc1",
    "waist_pc2": "state/waist/position_pc2",
}


def normalize_agibot_targets(targets):
    if isinstance(targets, dict):
        target_type = str(targets.get("type", "bimanual_compact_v1"))
        if target_type not in TARGET_PRESETS:
            raise ValueError(
                f"Unsupported AgiBot target preset {target_type!r}; "
                f"expected one of {sorted(TARGET_PRESETS)}"
            )
        target_list = list(TARGET_PRESETS[target_type])
        if target_type == "joint_effector_v1":
            joint_topk = int(targets.get("joint_topk", 4))
            target_list = []
            target_list += [f"state/joint/position[{i}]" for i in range(joint_topk)]
            target_list += [f"state/joint/velocity[{i}]" for i in range(joint_topk)]
            if bool(targets.get("include_end_velocity_norm", True)):
                target_list.append("state/end/velocity_norm")
            if bool(targets.get("include_effector_scalar", True)):
                target_list += [
                    "state/left_effector/position",
                    "state/right_effector/position",
                ]
        return target_list
    if isinstance(targets, str):
        return [t.strip() for t in targets.split(",") if t.strip()]
    return list(targets or TARGET_PRESETS["bimanual_compact_v1"])


def normalize_agibot_target_field(field):
    field = field.strip()
    return AGIBOT_TARGET_ALIASES.get(field, field)


def _parse_indexed_field(field):
    if "[" not in field or not field.endswith("]"):
        return field, None
    base, index_text = field[:-1].split("[", 1)
    return base, int(index_text)


def _field_value_from_state(state, field):
    field = normalize_agibot_target_field(field)
    if is_agibot_series_derived_field(field):
        raise ValueError(
            f"AgiBot field {field!r} is sequence-derived; use load_agibot_series"
        )
    if field == "state/end/velocity_norm":
        start, end = STATE_SLICES["state/end/velocity"]
        value = np.linalg.norm(state[start:end])
        return value, np.isfinite(value)

    derived_spec = AGIBOT_DERIVED_TARGETS.get(field)
    if derived_spec is not None:
        base_field, offset_start, offset_end = derived_spec
        base_start, _ = STATE_SLICES[base_field]
        value = np.linalg.norm(state[base_start + offset_start : base_start + offset_end])
        return value, np.isfinite(value)

    base, index = _parse_indexed_field(field)
    if base not in STATE_SLICES:
        raise ValueError(f"Unsupported AgiBot target field: {field}")
    start, end = STATE_SLICES[base]
    if index is None:
        values = state[start:end]
        if values.size != 1:
            raise ValueError(f"AgiBot target field {field!r} is vector-valued; add [index]")
        value = values[0]
    else:
        if index < 0 or index >= (end - start):
            raise ValueError(f"AgiBot target field index out of range: {field}")
        value = state[start + index]
    return value, np.isfinite(value)


def _list_to_array(values, min_dim=None):
    arr = np.asarray(values, dtype=np.float32)
    if min_dim is not None and arr.shape[-1] < min_dim:
        raise ValueError(f"Expected AgiBot state dim at least {min_dim}, got shape {arr.shape}")
    return arr


@lru_cache(maxsize=256)
def load_agibot_series(parquet_path, target_fields_tuple):
    target_fields = tuple(normalize_agibot_target_field(field) for field in target_fields_tuple)
    df = pd.read_parquet(parquet_path, columns=["timestamp", "observation.state"])
    if len(df) == 0:
        raise RuntimeError(f"Empty AgiBot parquet: {parquet_path}")

    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    states = np.stack(
        [_list_to_array(v, min_dim=AGIBOT_STATE_MIN_LEN) for v in df["observation.state"]],
        axis=0,
    )
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    states = states[order]
    if any(is_agibot_series_derived_field(field) for field in target_fields):
        if states.shape[-1] < SERIES_DERIVED_MIN_STATE_LEN:
            raise RuntimeError(
                f"AgiBot sequence-derived targets require observation.state dim at least "
                f"{SERIES_DERIVED_MIN_STATE_LEN}, got {states.shape[-1]} in {parquet_path}"
            )

    values = np.zeros((states.shape[0], len(target_fields)), dtype=np.float32)
    valid = np.zeros_like(values, dtype=np.float32)
    for col_idx, field in enumerate(target_fields):
        if is_agibot_series_derived_field(field):
            field_values, field_valid = derive_agibot_series_field(timestamps, states, field)
            values[:, col_idx] = field_values
            valid[:, col_idx] = field_valid
            continue
        for row_idx, state in enumerate(states):
            value, is_valid = _field_value_from_state(state, field)
            if is_valid:
                values[row_idx, col_idx] = float(value)
                valid[row_idx, col_idx] = 1.0

    return timestamps, values, valid


def build_tubelet_targets_from_times(
    parquet_path,
    sample_times,
    target_fields,
    tubelet_size,
    max_time_delta=None,
):
    timestamps, values, valid = load_agibot_series(str(parquet_path), tuple(target_fields))
    state_indices, time_valid = nearest_time_indices(sample_times, timestamps, max_delta=max_time_delta)
    per_frame_target = torch.from_numpy(values[state_indices])
    per_frame_valid = torch.from_numpy(valid[state_indices] * time_valid[:, None])
    if per_frame_target.size(0) % tubelet_size != 0:
        raise ValueError(
            f"Number of sampled frames must be divisible by tubelet_size; "
            f"got {per_frame_target.size(0)} and {tubelet_size}"
        )
    target = per_frame_target.reshape(-1, tubelet_size, per_frame_target.size(-1))
    valid_mask = per_frame_valid.reshape(-1, tubelet_size, per_frame_valid.size(-1))
    valid_sum = valid_mask.sum(dim=1).clamp_min(1.0)
    target = (target * valid_mask).sum(dim=1) / valid_sum
    valid_mask = (valid_mask.sum(dim=1) > 0).float()
    return target.float(), valid_mask.float()


def _feature_shape(info, video_key):
    feature = info.get("features", {}).get(video_key, {})
    shape = feature.get("shape") or []
    if len(shape) < 2:
        return None, None
    return int(shape[0]), int(shape[1])


def _feature_video_fps(info, video_key):
    feature = info.get("features", {}).get(video_key, {})
    video_info = feature.get("video_info", {})
    return float(video_info.get("video.fps", info.get("fps", 30)))


def _feature_video_codec(info, video_key):
    feature = info.get("features", {}).get(video_key, {})
    video_info = feature.get("video_info", {})
    return video_info.get("video.codec")


def agibot_episode_key(task_id, episode_index, session_id=None):
    episode = f"episode_{int(episode_index):06d}"
    if session_id:
        return f"task_{task_id}/{session_id}/{episode}"
    return f"task_{task_id}/{episode}"


def agibot_window_key(task_id, episode_index, window_idx, session_id=None):
    return f"{agibot_episode_key(task_id, episode_index, session_id)}/window_{int(window_idx):06d}"


def _is_window_key(key):
    return "/window_" in str(key)


def _parent_episode_key(window_key):
    return str(window_key).rsplit("/window_", 1)[0]


def _normalize_allowlists(episode_allowlist=None, sample_allowlist=None):
    episode_keys = set()
    window_keys = set()
    requested = set()
    for keys in (episode_allowlist, sample_allowlist):
        if keys is None:
            continue
        for key in keys:
            key = str(key)
            requested.add(key)
            if _is_window_key(key):
                window_keys.add(key)
                episode_keys.add(_parent_episode_key(key))
            else:
                episode_keys.add(key)
    return (
        episode_keys if episode_keys else None,
        window_keys if window_keys else None,
        requested if requested else None,
    )


def _normalize_view_allowlists(video_keys, view_episode_allowlists=None):
    if view_episode_allowlists is None:
        return None, None, None, False
    if not isinstance(view_episode_allowlists, dict):
        raise ValueError(
            "AgiBot view_episode_allowlists must be a mapping from video key "
            "to episode/window keys"
        )

    normalized_input = {
        str(video_key): keys
        for video_key, keys in view_episode_allowlists.items()
    }
    expected_views = set(video_keys)
    configured_views = set(normalized_input)
    if configured_views != expected_views:
        raise ValueError(
            "AgiBot view_episode_allowlists keys must exactly match video_keys; "
            f"missing={sorted(expected_views - configured_views)} "
            f"extra={sorted(configured_views - expected_views)}"
        )

    normalized = {}
    requested_pairs = set()
    episode_union = set()
    has_window_keys = False
    for video_key in video_keys:
        episode_keys, window_keys, requested_keys = _normalize_allowlists(
            episode_allowlist=normalized_input[video_key],
        )
        if requested_keys is None:
            raise ValueError(
                "AgiBot view_episode_allowlists entries must not be empty; "
                f"video_key={video_key}"
            )
        normalized[video_key] = (episode_keys, window_keys)
        episode_union.update(episode_keys or ())
        requested_pairs.update(
            (video_key, requested_key)
            for requested_key in requested_keys
        )
        has_window_keys = has_window_keys or window_keys is not None
    return normalized, requested_pairs, episode_union, has_window_keys


def _discover_task_data_dirs(root, task_id):
    task_root = pathlib.Path(root) / f"task_{task_id}"
    old_style = task_root / "data"
    if (old_style / "meta" / "info.json").exists():
        return [(None, old_style)]

    data_dirs = []
    for data_dir in sorted(task_root.glob("*/data")):
        if (data_dir / "meta" / "info.json").exists():
            data_dirs.append((data_dir.parent.name, data_dir))
    return data_dirs


def _read_ffmpeg_frames(video_path, frame_indices, height, width, timeout=120, seek_time=None):
    unique_indices, inverse = np.unique(np.asarray(frame_indices, dtype=np.int64), return_inverse=True)
    select_expr = "+".join(f"eq(n\\,{int(idx)})" for idx in unique_indices.tolist())
    frame_bytes = int(height) * int(width) * 3
    expected_bytes = len(unique_indices) * frame_bytes
    cmd = [
        os.environ.get("FFMPEG_BINARY", "ffmpeg"),
        "-v",
        "error",
    ]
    if seek_time is not None and float(seek_time) > 0:
        cmd += ["-ss", f"{float(seek_time):.6f}"]
    cmd += [
        "-i",
        str(video_path),
        "-vf",
        f"select={select_expr}",
        "-vsync",
        "0",
        "-frames:v",
        str(len(unique_indices)),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=None if timeout is None else float(timeout),
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}: {stderr}")
    if len(proc.stdout) != expected_bytes:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg returned {len(proc.stdout)} bytes, expected {expected_bytes}; stderr={stderr}"
        )
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(len(unique_indices), height, width, 3)
    return frames[inverse].copy()


def read_agibot_rows(
    root,
    tasks,
    video_key=None,
    video_keys=None,
    use_meta_episodes_only=True,
    max_windows_per_episode=None,
    frames_per_clip=64,
    fps=5,
    slice_mode="episode",
    slice_stride_seconds=None,
    episode_allowlist=None,
    sample_allowlist=None,
    view_episode_allowlists=None,
):
    root = pathlib.Path(root).expanduser().resolve()
    if video_keys is None:
        if video_key is None:
            video_keys = ["observation.images.top_head"]
        elif isinstance(video_key, (list, tuple)):
            video_keys = [str(key) for key in video_key]
        else:
            video_keys = [str(video_key)]
    else:
        if isinstance(video_keys, str):
            video_keys = [video_keys]
        else:
            video_keys = [str(key) for key in video_keys]
    allowlist, window_allowlist, requested_allowlist = _normalize_allowlists(
        episode_allowlist=episode_allowlist,
        sample_allowlist=sample_allowlist,
    )
    (
        normalized_view_allowlists,
        requested_view_pairs,
        view_episode_union,
        view_has_window_keys,
    ) = _normalize_view_allowlists(
        video_keys,
        view_episode_allowlists=view_episode_allowlists,
    )
    if allowlist is None and view_episode_union:
        allowlist = view_episode_union
    if window_allowlist is not None or view_has_window_keys:
        slice_mode = "sliding"
    rows = []
    matched_keys = set()
    matched_view_pairs = set()
    for task in tasks:
        task_id = str(task)
        task_data_dirs = _discover_task_data_dirs(root, task_id)
        if not task_data_dirs:
            raise FileNotFoundError(
                f"No AgiBot data directories found for task_{task_id} under {root}; "
                "expected task_<id>/data or task_<id>/<session>/data"
            )
        for session_id, task_dir in task_data_dirs:
            info_path = task_dir / "meta" / "info.json"
            episodes_path = task_dir / "meta" / "episodes.jsonl"
            if not episodes_path.exists():
                raise FileNotFoundError(f"AgiBot episodes.jsonl not found: {episodes_path}")
            with info_path.open("r") as f:
                info = json.load(f)
            feature_meta = {}
            for current_video_key in video_keys:
                if current_video_key not in info.get("features", {}):
                    raise ValueError(f"AgiBot video_key {current_video_key!r} not found in {info_path}")
                raw_fps = _feature_video_fps(info, current_video_key)
                height, width = _feature_shape(info, current_video_key)
                codec = _feature_video_codec(info, current_video_key)
                feature_meta[current_video_key] = {
                    "raw_fps": raw_fps,
                    "height": height,
                    "width": width,
                    "codec": codec,
                }
            data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
            video_template = info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
            episode_records = []
            with episodes_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    episode_records.append(json.loads(line))
            if not use_meta_episodes_only:
                chunk_dir = task_dir / "data" / "chunk-000"
                known = {int(e["episode_index"]) for e in episode_records}
                for parquet_path in sorted(chunk_dir.glob("episode_*.parquet")):
                    episode_index = int(parquet_path.stem.split("_")[-1])
                    if episode_index not in known:
                        episode_records.append({"episode_index": episode_index, "length": None})

            for record in episode_records:
                episode_index = int(record["episode_index"])
                episode_key = agibot_episode_key(task_id, episode_index, session_id)
                if allowlist is not None and episode_key not in allowlist:
                    continue

                episode_chunk = episode_index // int(info.get("chunks_size", 1000))
                parquet_rel = data_template.format(episode_chunk=episode_chunk, episode_index=episode_index)
                parquet_path = task_dir / parquet_rel
                if not parquet_path.exists():
                    continue
                length = record.get("length")
                if length is None:
                    try:
                        length = len(pd.read_parquet(parquet_path, columns=["timestamp"]))
                    except Exception:
                        length = 0
                for current_video_key in video_keys:
                    view_episode_allowlist = None
                    view_window_allowlist = None
                    if normalized_view_allowlists is not None:
                        (
                            view_episode_allowlist,
                            view_window_allowlist,
                        ) = normalized_view_allowlists[current_video_key]
                        if (
                            view_episode_allowlist is not None
                            and episode_key not in view_episode_allowlist
                        ):
                            continue
                    video_rel = video_template.format(
                        episode_chunk=episode_chunk,
                        episode_index=episode_index,
                        video_key=current_video_key,
                    )
                    video_path = task_dir / video_rel
                    if not video_path.exists():
                        continue
                    video_meta = feature_meta[current_video_key]
                    duration = max(0.0, (float(length) - 1.0) / float(video_meta["raw_fps"])) if length else 0.0
                    base_row = {
                        "domain": "agibot",
                        "task_id": task_id,
                        "session_id": session_id,
                        "episode_index": episode_index,
                        "episode_key": episode_key,
                        "video_key": current_video_key,
                        "video_path": str(video_path),
                        "parquet_path": str(parquet_path),
                        "length": int(length),
                        "raw_fps": float(video_meta["raw_fps"]),
                        "height": video_meta["height"],
                        "width": video_meta["width"],
                        "codec": video_meta["codec"],
                        "duration": duration,
                    }
                    if slice_mode == "sliding":
                        stride = float(slice_stride_seconds) if slice_stride_seconds is not None else float(frames_per_clip) / float(fps)
                        max_start = max_start_time(duration, frames_per_clip, fps)
                        starts = np.arange(0.0, max_start + 1e-6, stride, dtype=np.float64)
                        if starts.size == 0:
                            starts = np.array([0.0], dtype=np.float64)
                        if max_windows_per_episode is not None and max_windows_per_episode > 0:
                            starts = starts[: int(max_windows_per_episode)]
                        for window_idx, start_time in enumerate(starts.tolist()):
                            sample_key = agibot_window_key(task_id, episode_index, window_idx, session_id)
                            if window_allowlist is not None and sample_key not in window_allowlist:
                                continue
                            if (
                                view_window_allowlist is not None
                                and sample_key not in view_window_allowlist
                            ):
                                continue
                            row = dict(base_row)
                            row["window_idx"] = window_idx
                            row["start_time"] = float(start_time)
                            row["sample_key"] = sample_key
                            rows.append(row)
                            matched_keys.add(episode_key)
                            matched_keys.add(sample_key if window_allowlist is not None else episode_key)
                            matched_view_pairs.add((current_video_key, episode_key))
                            matched_view_pairs.add(
                                (
                                    current_video_key,
                                    sample_key
                                    if view_window_allowlist is not None
                                    else episode_key,
                                )
                            )
                    else:
                        sample_key = agibot_window_key(task_id, episode_index, 0, session_id)
                        if window_allowlist is not None and sample_key not in window_allowlist:
                            continue
                        if (
                            view_window_allowlist is not None
                            and sample_key not in view_window_allowlist
                        ):
                            continue
                        row = dict(base_row)
                        row["window_idx"] = 0
                        row["start_time"] = None
                        row["sample_key"] = sample_key
                        rows.append(row)
                        matched_keys.add(episode_key)
                        matched_keys.add(sample_key if window_allowlist is not None else episode_key)
                        matched_view_pairs.add((current_video_key, episode_key))
                        matched_view_pairs.add(
                            (
                                current_video_key,
                                sample_key
                                if view_window_allowlist is not None
                                else episode_key,
                            )
                        )
    if not rows:
        raise RuntimeError(f"No AgiBot rows found from root={root} tasks={tasks} video_keys={video_keys}")
    if requested_allowlist is not None:
        missing = sorted(requested_allowlist - matched_keys)
        if missing:
            raise RuntimeError(
                f"{len(missing)} keys from AgiBot allowlist were not found on disk for "
                f"tasks={tasks} video_keys={video_keys}; first={missing[:5]}"
            )
    if requested_view_pairs is not None:
        missing_view_pairs = sorted(requested_view_pairs - matched_view_pairs)
        if missing_view_pairs:
            raise RuntimeError(
                f"{len(missing_view_pairs)} view/key pairs from AgiBot per-view "
                "allowlists were not found on disk; "
                f"first={missing_view_pairs[:5]}"
            )
    return rows


class AgiBotTimeJEPADataset(torch.utils.data.Dataset):
    """AgiBot video dataset with timestamp-aligned tubelet-level robot targets."""

    def __init__(
        self,
        root,
        tasks=(3641, 3777),
        video_key="observation.images.top_head",
        video_keys=None,
        frames_per_clip=64,
        fps=5,
        target_fields=None,
        tubelet_size=2,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        slice_mode="random",
        slice_stride_seconds=None,
        max_windows_per_episode=None,
        use_meta_episodes_only=True,
        max_time_delta=None,
        video_backend="auto",
        ffmpeg_timeout=120,
        filter_short_videos=False,
        return_physical_targets=True,
        episode_allowlist=None,
        view_episode_allowlists=None,
    ):
        if fps is None:
            raise ValueError("AgiBotTimeJEPADataset requires data fps for timestamp sampling")
        self.root = pathlib.Path(root).expanduser().resolve()
        self.tasks = [str(t) for t in tasks]
        if video_keys is None:
            if isinstance(video_key, (list, tuple)):
                video_keys = [str(key) for key in video_key]
            else:
                video_keys = [str(video_key)]
        else:
            video_keys = [str(key) for key in (video_keys if isinstance(video_keys, (list, tuple)) else [video_keys])]
        self.video_keys = video_keys
        self.video_key = self.video_keys[0]
        self.frames_per_clip = int(frames_per_clip)
        self.fps = float(fps)
        self.target_fields = normalize_agibot_targets(target_fields)
        self.tubelet_size = int(tubelet_size)
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = bool(random_clip_sampling)
        self.slice_mode = str(slice_mode or "random").lower()
        self.max_time_delta = max_time_delta
        self.video_backend = str(video_backend or "auto").lower()
        if self.video_backend not in ("auto", "decord", "ffmpeg"):
            raise ValueError("AgiBot video_backend must be one of: auto, decord, ffmpeg")
        self.ffmpeg_timeout = ffmpeg_timeout
        self.filter_short_videos = bool(filter_short_videos)
        self.return_physical_targets = bool(return_physical_targets)
        has_window_allowlist = any(
            _is_window_key(key)
            for key in (episode_allowlist or [])
        ) or any(
            _is_window_key(key)
            for keys in (view_episode_allowlists or {}).values()
            for key in keys
        )
        manifest_slice_mode = "sliding" if self.slice_mode == "sliding" or has_window_allowlist else "episode"
        self.rows = read_agibot_rows(
            root=self.root,
            tasks=self.tasks,
            video_keys=self.video_keys,
            use_meta_episodes_only=use_meta_episodes_only,
            max_windows_per_episode=max_windows_per_episode,
            frames_per_clip=self.frames_per_clip,
            fps=self.fps,
            slice_mode=manifest_slice_mode,
            slice_stride_seconds=slice_stride_seconds,
            episode_allowlist=episode_allowlist,
            view_episode_allowlists=view_episode_allowlists,
        )
        if self.filter_short_videos:
            min_duration = (self.frames_per_clip - 1) / self.fps
            self.rows = [row for row in self.rows if row["duration"] >= min_duration]
        if not self.rows:
            raise RuntimeError("AgiBot dataset is empty after filtering")
        self.samples = [row["video_path"] for row in self.rows]
        self.labels = [0] * len(self.rows)
        self.sample_weights = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        loaded_sample = False
        while not loaded_sample:
            loaded_sample = self.get_item_video(index)
            if not loaded_sample:
                index = np.random.randint(self.__len__())
                row = self.rows[index]
        return loaded_sample

    def _sample_start_time(self, row):
        if row.get("start_time") is not None:
            return float(row["start_time"])
        max_start = max_start_time(row["duration"], self.frames_per_clip, self.fps)
        if self.random_clip_sampling and max_start > 0:
            return float(np.random.uniform(0.0, max_start))
        return 0.0

    def _read_decord_clip(self, video_path, sample_times, row):
        vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        if len(vr) <= 0:
            raise RuntimeError("empty decord video")
        video_fps = infer_video_fps(vr, fallback=row.get("raw_fps", 30.0))
        frame_indices = times_to_frame_indices(sample_times, video_fps=video_fps, num_frames=len(vr))
        return vr.get_batch(frame_indices.tolist()).asnumpy(), frame_indices, "decord"

    def _read_ffmpeg_clip(self, video_path, sample_times, row, start_time):
        height = row.get("height")
        width = row.get("width")
        if height is None or width is None:
            raise RuntimeError(f"AgiBot row is missing video shape metadata: {video_path}")
        video_fps = float(row.get("raw_fps", 30.0))
        num_frames = int(row.get("length", 0))
        if num_frames <= 0:
            raise RuntimeError(f"AgiBot row has invalid frame count: {video_path}")
        frame_indices = times_to_frame_indices(sample_times, video_fps=video_fps, num_frames=num_frames)
        first_frame_idx = int(frame_indices[0])
        relative_indices = frame_indices - first_frame_idx
        seek_time = float(first_frame_idx) / video_fps
        buffer = _read_ffmpeg_frames(
            video_path=video_path,
            frame_indices=relative_indices,
            height=int(height),
            width=int(width),
            timeout=self.ffmpeg_timeout,
            seek_time=seek_time,
        )
        return buffer, frame_indices, "ffmpeg"

    def _read_video_clip(self, video_path, sample_times, row, start_time):
        if self.video_backend in ("auto", "decord"):
            try:
                return self._read_decord_clip(video_path, sample_times, row)
            except Exception as exc:
                if self.video_backend == "decord":
                    raise
                warnings.warn(f"Decord failed for AgiBot video {video_path}; falling back to ffmpeg: {exc}")
        return self._read_ffmpeg_clip(video_path, sample_times, row, start_time)

    def get_item_video(self, index):
        row = self.rows[index]
        video_path = row["video_path"]
        if not os.path.exists(video_path):
            warnings.warn(f"AgiBot video path not found: {video_path}")
            return
        try:
            start_time = self._sample_start_time(row)
            sample_times = build_sample_times(start_time, self.frames_per_clip, self.fps)
            buffer, frame_indices, backend_used = self._read_video_clip(
                video_path,
                sample_times,
                row,
                start_time,
            )
            if self.return_physical_targets:
                target, valid = build_tubelet_targets_from_times(
                    parquet_path=row["parquet_path"],
                    sample_times=sample_times,
                    target_fields=self.target_fields,
                    tubelet_size=self.tubelet_size,
                    max_time_delta=self.max_time_delta,
                )
        except Exception as exc:
            warnings.warn(f"Failed to build AgiBot sample {video_path}: {exc}")
            return

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        if self.transform is not None:
            buffer = [self.transform(buffer)]
        else:
            buffer = [buffer]

        if not self.return_physical_targets:
            return buffer, 0, [frame_indices]

        physical = {
            "domain": "agibot",
            "target": [target],
            "valid": [valid],
            "video_id": f"{row.get('sample_key', row.get('episode_key'))}/{row['video_key']}",
            "task_id": row["task_id"],
            "session_id": row.get("session_id", ""),
            "episode_index": row["episode_index"],
            "video_key": row["video_key"],
            "video_backend": backend_used,
            "start_time": float(start_time),
            "sample_times": [torch.from_numpy(sample_times.astype(np.float32))],
        }
        return buffer, 0, [frame_indices], physical


def make_agibot_time_jepa_dataset(
    data_paths,
    batch_size,
    frames_per_clip=64,
    dataset_fpcs=None,
    frame_step=None,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    agibot_tasks=None,
    agibot_video_key="observation.images.top_head",
    agibot_video_keys=None,
    agibot_target_fields=None,
    physical_tubelet_size=2,
    agibot_slice_mode="random",
    agibot_slice_stride_seconds=None,
    agibot_max_windows_per_episode=None,
    agibot_use_meta_episodes_only=True,
    agibot_max_time_delta=None,
    agibot_video_backend="auto",
    agibot_ffmpeg_timeout=120,
    return_physical_targets=True,
    agibot_episode_allowlist=None,
    agibot_view_episode_allowlists=None,
):
    if num_clips != 1:
        raise NotImplementedError("AgiBotTimeJEPADataset currently supports num_clips=1")
    if dataset_fpcs is not None:
        if len(dataset_fpcs) != 1:
            raise NotImplementedError("AgiBotTimeJEPADataset currently expects one frames-per-clip value")
        frames_per_clip = int(dataset_fpcs[0])
    if isinstance(data_paths, (list, tuple)):
        if len(data_paths) != 1:
            raise ValueError("AgiBotTimeJEPADataset expects a single root path")
        root = data_paths[0]
    else:
        root = data_paths

    dataset = AgiBotTimeJEPADataset(
        root=root,
        tasks=agibot_tasks or (3641, 3777),
        video_key=agibot_video_key,
        video_keys=agibot_video_keys,
        frames_per_clip=frames_per_clip,
        fps=fps,
        target_fields=agibot_target_fields,
        tubelet_size=physical_tubelet_size,
        transform=transform,
        shared_transform=shared_transform,
        random_clip_sampling=random_clip_sampling,
        slice_mode=agibot_slice_mode,
        slice_stride_seconds=agibot_slice_stride_seconds,
        max_windows_per_episode=agibot_max_windows_per_episode,
        use_meta_episodes_only=agibot_use_meta_episodes_only,
        max_time_delta=agibot_max_time_delta,
        video_backend=agibot_video_backend,
        ffmpeg_timeout=agibot_ffmpeg_timeout,
        filter_short_videos=filter_short_videos,
        return_physical_targets=return_physical_targets,
        episode_allowlist=agibot_episode_allowlist,
        view_episode_allowlists=agibot_view_episode_allowlists,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info(
        "AgiBotTimeJEPADataset created rows=%s targets=%s video_key=%s",
        len(dataset),
        getattr(dataset, "target_fields", agibot_target_fields),
        agibot_video_key,
    )
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    loader_cls = torch.utils.data.DataLoader if deterministic else NondeterministicDataLoader
    data_loader = loader_cls(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("AgiBotTimeJEPADataset unsupervised data loader created")
    return dataset, data_loader, dist_sampler


def compute_agibot_stats(
    rows,
    target_fields,
    frames_per_clip,
    tubelet_size,
    fps,
    seed=0,
    max_videos=500,
    max_time_delta=None,
):
    rows = list(rows)
    rng = np.random.RandomState(seed)
    if max_videos > 0 and len(rows) > max_videos:
        rows = [rows[i] for i in rng.choice(len(rows), size=max_videos, replace=False).tolist()]
    target_fields = normalize_agibot_targets(target_fields)
    sums = torch.zeros(len(target_fields), dtype=torch.float64)
    sq_sums = torch.zeros(len(target_fields), dtype=torch.float64)
    counts = torch.zeros(len(target_fields), dtype=torch.float64)
    skipped_errors = []
    successful_rows = 0
    for row in rows:
        try:
            start_time = row.get("start_time")
            if start_time is None:
                start_time = 0.0
                max_start = max_start_time(row["duration"], frames_per_clip, fps)
                if max_start > 0:
                    start_time = float(rng.uniform(0.0, max_start))
            sample_times = build_sample_times(start_time, frames_per_clip, fps)
            target, valid = build_tubelet_targets_from_times(
                parquet_path=row["parquet_path"],
                sample_times=sample_times,
                target_fields=target_fields,
                tubelet_size=tubelet_size,
                max_time_delta=max_time_delta,
            )
        except Exception as exc:
            if len(skipped_errors) < 5:
                skipped_errors.append(
                    f"{row.get('parquet_path', '<unknown>')}: "
                    f"{type(exc).__name__}: {exc}"
                )
            continue
        target = target.double()
        valid = valid.double()
        sums += (target * valid).sum(dim=0)
        sq_sums += ((target**2) * valid).sum(dim=0)
        counts += valid.sum(dim=0)
        successful_rows += 1
    if (counts < 2).any():
        missing = [target_fields[i] for i, count in enumerate(counts.tolist()) if count < 2]
        error_hint = "; ".join(skipped_errors) if skipped_errors else "no row-level exceptions captured"
        raise RuntimeError(
            f"Could not compute AgiBot target stats for fields: {missing}. "
            f"rows={len(rows)} successful_rows={successful_rows} counts={counts.tolist()}. "
            f"First skipped errors: {error_hint}"
        )
    mean = sums / counts
    var = (sq_sums - (sums * sums) / counts) / (counts - 1.0).clamp_min(1.0)
    std = torch.sqrt(var.clamp_min(1e-12)).clamp_min(1e-6)
    return mean.float(), std.float(), counts.long()
