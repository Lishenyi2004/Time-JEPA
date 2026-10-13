"""Create fixed AgiBot window splits for ID top_head and held-out-view OOD.

The updated AgiBot root is sessioned:

    task_<id>/<session_range>/data/
      meta/{info.json,episodes.jsonl}
      data/chunk-000/episode_*.parquet
      videos/chunk-000/<video_key>/episode_*.mp4

Split keys are camera-agnostic window keys:

    task_<id>/<session_range>/episode_<idx>/window_<idx>

This keeps train/probe/test episode-disjoint while letting ID evaluation use
``top_head`` and OOD evaluation reuse the same held-out windows with
``hand_left`` / ``hand_right``.
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_TASKS = ["3400", "3401", "3402", "3404", "3405", "3477", "3641", "3705", "3777", "4053"]
DEFAULT_SPLITS = ["encoder_train=7000", "probe_train=1500", "probe_val=500", "test=1000"]
DEFAULT_ID_VIEW = "observation.images.top_head"
DEFAULT_OOD_VIEWS = ["observation.images.hand_left", "observation.images.hand_right"]


def episode_key(task_id, session_id, episode_index):
    return f"task_{task_id}/{session_id}/episode_{int(episode_index):06d}"


def window_key(task_id, session_id, episode_index, window_idx):
    return f"{episode_key(task_id, session_id, episode_index)}/window_{int(window_idx):06d}"


def parse_named_counts(items):
    splits = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Split must use name=count format, got: {item}")
        name, count_text = item.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid split name {name!r}; use letters, numbers, '_' or '-'")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"Split count must be non-negative, got {item}")
        splits.append((name, count))
    if not splits:
        raise ValueError("At least one split is required")
    if sum(count for _, count in splits) <= 0:
        raise ValueError("Split counts must sum to a positive value")
    return splits


def parse_named_ratios(items):
    splits = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Split must use name=ratio format, got: {item}")
        name, ratio_text = item.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid split name {name!r}; use letters, numbers, '_' or '-'")
        ratio = float(ratio_text)
        if ratio < 0:
            raise ValueError(f"Split ratio must be non-negative, got {item}")
        splits.append((name, ratio))
    total = sum(ratio for _, ratio in splits)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    return splits


def discover_task_data_dirs(agibot_root, task_id):
    task_root = agibot_root / f"task_{task_id}"
    old_style = task_root / "data"
    if (old_style / "meta" / "info.json").exists():
        return [(None, old_style)]
    return [
        (data_dir.parent.name, data_dir)
        for data_dir in sorted(task_root.glob("*/data"))
        if (data_dir / "meta" / "info.json").exists()
    ]


def video_exists(data_dir, info, video_key, episode_index):
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = int(episode_index) // chunk_size
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    video_path = data_dir / video_template.format(
        episode_chunk=episode_chunk,
        episode_index=int(episode_index),
        video_key=video_key,
    )
    return video_path.exists()


def parquet_exists(data_dir, info, episode_index):
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = int(episode_index) // chunk_size
    data_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    parquet_path = data_dir / data_template.format(
        episode_chunk=episode_chunk,
        episode_index=int(episode_index),
    )
    return parquet_path.exists()


def episode_windows(length, raw_fps, frames_per_clip, fps, stride_seconds):
    if length is None or int(length) <= 0:
        return []
    duration = (float(length) - 1.0) / float(raw_fps)
    max_start = duration - (int(frames_per_clip) - 1) / float(fps)
    if max_start < 0:
        return []
    starts = []
    start = 0.0
    while start <= max_start + 1e-6:
        starts.append(start)
        start += float(stride_seconds)
    return starts


def read_episode_records(agibot_root, tasks, id_view, ood_views, frames_per_clip, fps, stride_seconds):
    per_task = defaultdict(list)
    skipped = defaultdict(int)
    view_presence = defaultdict(int)
    for task in tasks:
        task_id = str(task)
        task_data_dirs = discover_task_data_dirs(agibot_root, task_id)
        if not task_data_dirs:
            raise FileNotFoundError(
                f"No AgiBot data directories found for task_{task_id} under {agibot_root}; "
                "expected task_<id>/data or task_<id>/<session>/data"
            )
        for session_id, data_dir in task_data_dirs:
            session_key = session_id or "data"
            info_path = data_dir / "meta" / "info.json"
            episodes_path = data_dir / "meta" / "episodes.jsonl"
            with info_path.open("r") as f:
                info = json.load(f)
            if id_view not in info.get("features", {}):
                skipped["missing_id_view_feature"] += 1
                continue
            raw_fps = float(
                info.get("features", {})
                .get(id_view, {})
                .get("video_info", {})
                .get("video.fps", info.get("fps", 30.0))
            )
            with episodes_path.open("r") as f:
                episode_rows = [json.loads(line) for line in f if line.strip()]
            for record in episode_rows:
                episode_index = int(record["episode_index"])
                if not parquet_exists(data_dir, info, episode_index):
                    skipped["missing_parquet"] += 1
                    continue
                if not video_exists(data_dir, info, id_view, episode_index):
                    skipped["missing_id_view_video"] += 1
                    continue
                missing_ood_view = False
                for ood_view in ood_views:
                    if not video_exists(data_dir, info, ood_view, episode_index):
                        skipped[f"missing_ood_view_video:{ood_view}"] += 1
                        missing_ood_view = True
                if missing_ood_view:
                    continue
                for view in [id_view] + list(ood_views):
                    if video_exists(data_dir, info, view, episode_index):
                        view_presence[view] += 1
                starts = episode_windows(
                    length=record.get("length"),
                    raw_fps=raw_fps,
                    frames_per_clip=frames_per_clip,
                    fps=fps,
                    stride_seconds=stride_seconds,
                )
                if not starts:
                    skipped["too_short"] += 1
                    continue
                key = episode_key(task_id, session_key, episode_index)
                per_task[task_id].append(
                    {
                        "episode_key": key,
                        "task_id": task_id,
                        "session_id": session_key,
                        "episode_index": episode_index,
                        "length": int(record.get("length") or 0),
                        "raw_fps": raw_fps,
                        "num_windows": len(starts),
                    }
                )
    return per_task, dict(skipped), dict(view_presence)


def split_episodes_stratified(per_task_episodes, named_ratios, seed):
    splits = {name: [] for name, _ in named_ratios}
    for task_id in sorted(per_task_episodes):
        episodes = list(per_task_episodes[task_id])
        rng = random.Random(f"{seed}:{task_id}:episodes")
        rng.shuffle(episodes)
        start = 0
        for idx, (name, ratio) in enumerate(named_ratios):
            end = len(episodes) if idx == len(named_ratios) - 1 else start + int(len(episodes) * ratio)
            splits[name].extend(episodes[start:end])
            start = end
    return splits


def sample_windows_from_episodes(split_episodes, target_count, seed, split_name):
    candidates = []
    for episode in split_episodes:
        for window_idx in range(int(episode["num_windows"])):
            candidates.append(
                window_key(
                    episode["task_id"],
                    episode["session_id"],
                    episode["episode_index"],
                    window_idx,
                )
            )
    if len(candidates) < target_count:
        raise RuntimeError(
            f"AgiBot split {split_name} has only {len(candidates)} windows, "
            f"but {target_count} were requested"
        )
    rng = random.Random(f"{seed}:{split_name}:windows")
    rng.shuffle(candidates)
    return sorted(candidates[:target_count]), len(candidates)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agibot-root",
        default="data/agibot",
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", default="time-jepa/splits")
    parser.add_argument("--prefix", default="agibot_top_head")
    parser.add_argument("--manifest-name", default="agibot_top_head_split_manifest.json")
    parser.add_argument("--id-view", default=DEFAULT_ID_VIEW)
    parser.add_argument("--ood-views", nargs="+", default=DEFAULT_OOD_VIEWS)
    parser.add_argument("--frames-per-clip", type=int, default=64)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--slice-stride-seconds", type=float, default=12.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episode-split-ratios",
        nargs="+",
        default=["encoder_train=0.70", "probe_train=0.15", "probe_val=0.05", "test=0.10"],
    )
    parser.add_argument(
        "--split-counts",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Named exact window counts, default mirrors CoVLA strict 70/15/5/10.",
    )
    args = parser.parse_args()

    agibot_root = Path(args.agibot_root).expanduser().resolve()
    tasks = [str(task) for task in args.tasks]
    named_ratios = parse_named_ratios(args.episode_split_ratios)
    named_counts = parse_named_counts(args.split_counts)
    ratio_names = [name for name, _ in named_ratios]
    count_names = [name for name, _ in named_counts]
    if ratio_names != count_names:
        raise ValueError(
            f"Episode ratio split names and count split names must match; got {ratio_names} vs {count_names}"
        )

    per_task_episodes, skipped, view_presence = read_episode_records(
        agibot_root=agibot_root,
        tasks=tasks,
        id_view=args.id_view,
        ood_views=args.ood_views,
        frames_per_clip=args.frames_per_clip,
        fps=args.fps,
        stride_seconds=args.slice_stride_seconds,
    )
    episode_splits = split_episodes_stratified(per_task_episodes, named_ratios, args.seed)

    output_dir = Path(args.output_dir)
    split_files = {}
    split_counts = {}
    split_capacity = {}
    split_episode_counts = {}
    for split_name, target_count in named_counts:
        keys, capacity = sample_windows_from_episodes(
            episode_splits[split_name],
            target_count=target_count,
            seed=args.seed,
            split_name=split_name,
        )
        split_path = output_dir / f"{args.prefix}_{split_name}.json"
        write_json(split_path, keys)
        split_files[split_name] = str(split_path.resolve())
        split_counts[split_name] = len(keys)
        split_capacity[split_name] = capacity
        split_episode_counts[split_name] = len(episode_splits[split_name])

    per_task_total = {
        task_id: {
            "episodes": len(episodes),
            "windows": sum(int(ep["num_windows"]) for ep in episodes),
        }
        for task_id, episodes in sorted(per_task_episodes.items())
    }
    manifest = {
        "agibot_root": str(agibot_root),
        "tasks": tasks,
        "seed": args.seed,
        "id_view": args.id_view,
        "ood_views": list(args.ood_views),
        "key_format": "task_{task_id}/{session_id}/episode_{episode_index:06d}/window_{window_idx:06d}",
        "frames_per_clip": args.frames_per_clip,
        "fps": args.fps,
        "slice_stride_seconds": args.slice_stride_seconds,
        "episode_split_ratios": {name: ratio for name, ratio in named_ratios},
        "requested_counts": {name: count for name, count in named_counts},
        "counts": split_counts,
        "capacity_windows": split_capacity,
        "episode_counts": split_episode_counts,
        "per_task_total": per_task_total,
        "view_presence_episodes": view_presence,
        "skipped": skipped,
        "files": split_files,
    }
    write_json(output_dir / args.manifest_name, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
