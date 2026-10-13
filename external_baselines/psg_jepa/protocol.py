"""Resolve the frozen AgiBot protocol into PSG-JEPA training records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml


UPSTREAM_REPOSITORY = "https://github.com/Haodong-Yan/PSG-JEPA"
EXPECTED_UPSTREAM_COMMIT = "3bf67a47a9143f9f4fb4d39f839143c92902714c"

DEFAULT_RAW_FPS = 30
DEFAULT_MODEL_FPS = 5
DEFAULT_WINDOW_FRAMES = 64
DEFAULT_MICROCLIP_FRAMES = 4
DEFAULT_ACTION_DIM = 40

_SAMPLE_KEY_RE = re.compile(
    r"^task_(?P<task>[^/]+)/"
    r"(?:(?P<session>[^/]+)/)?"
    r"episode_(?P<episode>[0-9]+)/window_(?P<window>[0-9]+)$"
)


@dataclass(frozen=True)
class AgiBotPSGRecord:
    """One physical 64-frame window from one camera view."""

    sample_key: str
    video_key: str
    task_id: str
    session_id: str | None
    episode_index: int
    window_index: int
    start_time: float
    parquet_path: Path
    video_path: Path
    raw_fps: float
    height: int
    width: int


@dataclass(frozen=True)
class AgiBotPSGProtocol:
    """Immutable subset and sampling contract shared by LeWM and PSG-static."""

    params_path: Path
    root: Path
    tasks: tuple[str, ...]
    video_keys: tuple[str, ...]
    sample_keys: tuple[str, ...]
    sample_seed: int
    window_frames: int
    model_fps: float
    raw_fps: float
    slice_stride_seconds: float
    max_time_delta: float
    state_stats_path: Path | None
    random_resize_scale: tuple[float, float]
    random_resize_aspect_ratio: tuple[float, float]
    # Newer Time-JEPA runs sample disjoint ``(view, window)`` pairs with a
    # separate quota for each camera.  ``None`` retains the legacy Cartesian
    # product used by the earlier 5,000-window / 15,000-view protocol.
    view_sample_keys: tuple[tuple[str, tuple[str, ...]], ...] | None = None

    @property
    def action_steps_per_model_step(self) -> int:
        ratio = self.raw_fps / self.model_fps
        rounded = int(round(ratio))
        if abs(ratio - rounded) > 1.0e-8:
            raise ValueError(
                "PSG action packing requires an integer raw/model FPS ratio; "
                f"got raw_fps={self.raw_fps}, model_fps={self.model_fps}"
            )
        return rounded

    @property
    def microclips_per_window(self) -> int:
        if self.window_frames % DEFAULT_MICROCLIP_FRAMES:
            raise ValueError(
                f"window_frames={self.window_frames} must be divisible by "
                f"{DEFAULT_MICROCLIP_FRAMES}"
            )
        return self.window_frames // DEFAULT_MICROCLIP_FRAMES

    @property
    def num_view_windows(self) -> int:
        if self.view_sample_keys is not None:
            return sum(len(sample_keys) for _, sample_keys in self.view_sample_keys)
        return len(self.sample_keys) * len(self.video_keys)

    def iter_records(self) -> Iterable[AgiBotPSGRecord]:
        if self.view_sample_keys is not None:
            sample_keys_by_view = dict(self.view_sample_keys)
            for video_key in self.video_keys:
                for sample_key in sample_keys_by_view[video_key]:
                    yield resolve_record(
                        self,
                        sample_key=sample_key,
                        video_key=video_key,
                    )
            return
        for sample_key in self.sample_keys:
            for video_key in self.video_keys:
                yield resolve_record(self, sample_key=sample_key, video_key=video_key)

    @classmethod
    def from_params(
        cls,
        params_path: str | Path,
        *,
        raw_fps: float = DEFAULT_RAW_FPS,
    ) -> "AgiBotPSGProtocol":
        params_path = Path(params_path).expanduser().resolve()
        if not params_path.is_file():
            raise FileNotFoundError(f"AgiBot resolved params file not found: {params_path}")
        with params_path.open("r") as handle:
            params = yaml.safe_load(handle) or {}

        data_cfg = _require_mapping(params, "data")
        time_cfg = _require_mapping(params, "time_jepa")
        domains_cfg = _require_mapping(time_cfg, "domains")
        domain_cfg = _require_mapping(domains_cfg, "agibot")

        root_value = domain_cfg.get("root")
        if not root_value:
            datasets = data_cfg.get("datasets") or []
            if len(datasets) != 1:
                raise ValueError(
                    "Resolved AgiBot params must provide time_jepa.domains.agibot.root "
                    "or exactly one data.datasets entry"
                )
            root_value = datasets[0]

        sample_keys = tuple(str(key) for key in domain_cfg.get("episode_allowlist") or ())
        if not sample_keys:
            raise ValueError(
                "Resolved AgiBot params contain no time_jepa.domains.agibot.episode_allowlist"
            )
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError("Resolved AgiBot allowlist contains duplicate physical windows")
        for key in sample_keys:
            parse_sample_key(key)

        video_value = domain_cfg.get("video_key")
        if isinstance(video_value, str):
            video_keys = (video_value,)
        else:
            video_keys = tuple(str(value) for value in (video_value or ()))
        if not video_keys:
            raise ValueError("Resolved AgiBot params contain no training video_key")
        if len(video_keys) != len(set(video_keys)):
            raise ValueError("Resolved AgiBot video_key list contains duplicates")

        view_allowlists_value = domain_cfg.get("view_episode_allowlists")
        view_sample_keys = None
        if view_allowlists_value is not None:
            if not isinstance(view_allowlists_value, dict):
                raise ValueError("view_episode_allowlists must be a mapping")
            missing_views = set(video_keys) - set(view_allowlists_value)
            extra_views = set(view_allowlists_value) - set(video_keys)
            if missing_views or extra_views:
                raise ValueError(
                    "view_episode_allowlists keys must exactly match video_key: "
                    f"missing={sorted(missing_views)}, extra={sorted(extra_views)}"
                )
            resolved_view_keys = []
            exposed_pairs = set()
            exposed_sample_keys = set()
            for video_key in video_keys:
                per_view_keys = tuple(
                    str(key) for key in (view_allowlists_value[video_key] or ())
                )
                if not per_view_keys:
                    raise ValueError(
                        f"view_episode_allowlists[{video_key!r}] is empty"
                    )
                if len(per_view_keys) != len(set(per_view_keys)):
                    raise ValueError(
                        f"view_episode_allowlists[{video_key!r}] contains duplicates"
                    )
                for key in per_view_keys:
                    parse_sample_key(key)
                    pair = (video_key, key)
                    if pair in exposed_pairs:
                        raise ValueError(f"Duplicate view-window pair: {pair}")
                    exposed_pairs.add(pair)
                    exposed_sample_keys.add(key)
                resolved_view_keys.append((video_key, per_view_keys))
            if exposed_sample_keys != set(sample_keys):
                raise ValueError(
                    "The union of view_episode_allowlists must match episode_allowlist: "
                    f"view_union={len(exposed_sample_keys)}, "
                    f"episode_allowlist={len(sample_keys)}"
                )
            configured_count = domain_cfg.get("encoder_train_view_num_samples")
            if configured_count is not None:
                if not isinstance(configured_count, dict):
                    raise ValueError("encoder_train_view_num_samples must be a mapping")
                actual_counts = {
                    video_key: len(per_view_keys)
                    for video_key, per_view_keys in resolved_view_keys
                }
                expected_counts = {
                    str(video_key): int(count)
                    for video_key, count in configured_count.items()
                }
                if actual_counts != expected_counts:
                    raise ValueError(
                        "Per-view allowlist sizes do not match configured quotas: "
                        f"actual={actual_counts}, expected={expected_counts}"
                    )
            view_sample_keys = tuple(resolved_view_keys)

        window_frames = int((data_cfg.get("dataset_fpcs") or [data_cfg.get("frames_per_clip", 64)])[0])
        model_fps = float(data_cfg.get("fps", DEFAULT_MODEL_FPS))
        slice_stride_seconds = float(
            domain_cfg.get("slice_stride_seconds", window_frames / model_fps)
        )
        max_time_delta = float(domain_cfg.get("max_time_delta", 0.5 / model_fps))
        tasks = tuple(str(task) for task in domain_cfg.get("tasks") or ())
        if not tasks:
            raise ValueError("Resolved AgiBot params contain no task list")

        state_stats_value = domain_cfg.get("stats_cache")
        data_aug = params.get("data_aug") or {}
        protocol = cls(
            params_path=params_path,
            root=Path(root_value).expanduser().resolve(),
            tasks=tasks,
            video_keys=video_keys,
            sample_keys=sample_keys,
            sample_seed=int(domain_cfg.get("encoder_train_sample_seed", 0)),
            window_frames=window_frames,
            model_fps=model_fps,
            raw_fps=float(raw_fps),
            slice_stride_seconds=slice_stride_seconds,
            max_time_delta=max_time_delta,
            state_stats_path=(
                Path(state_stats_value).expanduser().resolve()
                if state_stats_value
                else None
            ),
            random_resize_scale=_float_pair(
                data_aug.get("random_resize_scale", (0.3, 1.0)),
                "random_resize_scale",
            ),
            random_resize_aspect_ratio=_float_pair(
                data_aug.get("random_resize_aspect_ratio", (0.75, 1.35)),
                "random_resize_aspect_ratio",
            ),
            view_sample_keys=view_sample_keys,
        )
        if protocol.action_steps_per_model_step != 6:
            raise ValueError(
                "The audited AgiBot PSG protocol expects six 30 Hz actions per 5 Hz step; "
                f"got {protocol.action_steps_per_model_step}"
            )
        if protocol.window_frames != DEFAULT_WINDOW_FRAMES:
            raise ValueError(
                f"The exposure-matched PSG protocol requires {DEFAULT_WINDOW_FRAMES} frames, "
                f"got {protocol.window_frames}"
            )
        return protocol


def _float_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}")
    pair = (float(value[0]), float(value[1]))
    if pair[0] <= 0 or pair[1] < pair[0]:
        raise ValueError(f"Invalid {name}: {pair}")
    return pair


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at {key!r}")
    return value


def parse_sample_key(sample_key: str) -> tuple[str, str | None, int, int]:
    match = _SAMPLE_KEY_RE.fullmatch(str(sample_key))
    if match is None:
        raise ValueError(f"Unsupported AgiBot sample key: {sample_key!r}")
    return (
        match.group("task"),
        match.group("session"),
        int(match.group("episode")),
        int(match.group("window")),
    )


@lru_cache(maxsize=256)
def _load_info(info_path: str) -> dict[str, Any]:
    path = Path(info_path)
    if not path.is_file():
        raise FileNotFoundError(f"AgiBot info.json not found: {path}")
    with path.open("r") as handle:
        return json.load(handle)


def resolve_record(
    protocol: AgiBotPSGProtocol,
    *,
    sample_key: str,
    video_key: str,
) -> AgiBotPSGRecord:
    task_id, session_id, episode_index, window_index = parse_sample_key(sample_key)
    if task_id not in protocol.tasks:
        raise ValueError(f"Sample task {task_id} is outside the frozen task list")

    task_root = protocol.root / f"task_{task_id}"
    data_root = task_root / "data" if session_id is None else task_root / session_id / "data"
    info_path = data_root / "meta" / "info.json"
    info = _load_info(str(info_path))
    features = info.get("features") or {}
    if video_key not in features:
        raise KeyError(f"Video key {video_key!r} is missing from {info_path}")

    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    data_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    parquet_path = data_root / data_template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    video_path = data_root / video_template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )

    feature = features[video_key]
    shape = feature.get("shape") or ()
    if len(shape) < 2:
        raise ValueError(f"Missing image shape for {video_key!r} in {info_path}")
    video_info = feature.get("video_info") or {}
    feature_fps = float(video_info.get("video.fps", info.get("fps", protocol.raw_fps)))
    if abs(feature_fps - protocol.raw_fps) > 1.0e-6:
        raise ValueError(
            f"Unexpected AgiBot FPS for {sample_key}/{video_key}: "
            f"metadata={feature_fps}, expected={protocol.raw_fps}"
        )

    return AgiBotPSGRecord(
        sample_key=sample_key,
        video_key=video_key,
        task_id=task_id,
        session_id=session_id,
        episode_index=episode_index,
        window_index=window_index,
        start_time=float(window_index) * protocol.slice_stride_seconds,
        parquet_path=parquet_path,
        video_path=video_path,
        raw_fps=feature_fps,
        height=int(shape[0]),
        width=int(shape[1]),
    )
