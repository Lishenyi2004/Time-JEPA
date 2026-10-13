"""Fit the versioned AgiBot waist PCA using encoder-train windows only.

The covariance is computed after centering each 12.8-second window, then
averaged equally across tasks.  This makes the basis describe waist motion
rather than task/session offsets and prevents large tasks from dominating it.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_key(key):
    task_part, session_id, episode_part, window_part = str(key).split("/")
    return (
        task_part.removeprefix("task_"),
        session_id,
        int(episode_part.removeprefix("episode_")),
        int(window_part.removeprefix("window_")),
    )


def nearest_indices(query, source):
    right = np.searchsorted(source, query, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    return np.where(np.abs(source[left] - query) <= np.abs(source[right] - query), left, right)


def canonicalize_component_signs(components):
    components = components.copy()
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agibot-root",
        default="data/agibot",
    )
    parser.add_argument(
        "--split",
        default="time-jepa/splits/agibot_top_head_encoder_train.json",
    )
    parser.add_argument(
        "--output",
        default="time-jepa/cache/agibot_wholebody_cartesian_v1_waist_pca.json",
    )
    parser.add_argument("--max-windows-per-task", type=int, default=128)
    parser.add_argument("--frames-per-window", type=int, default=64)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--stride-seconds", type=float, default=12.8)
    parser.add_argument("--seed", type=int, default=239)
    args = parser.parse_args()

    split_path = Path(args.split).expanduser().resolve()
    split_keys = json.loads(split_path.read_text())
    by_task = defaultdict(list)
    for key in split_keys:
        by_task[parse_key(key)[0]].append(key)

    rng = np.random.default_rng(args.seed)
    task_raw_sums = {}
    task_raw_counts = {}
    task_centered_covariances = {}
    task_centered_counts = {}
    task_windows = {}
    root = Path(args.agibot_root).expanduser().resolve()

    for task_id, keys in sorted(by_task.items()):
        count = min(len(keys), int(args.max_windows_per_task))
        chosen = [keys[i] for i in rng.choice(len(keys), size=count, replace=False)]
        raw_sum = np.zeros(5, dtype=np.float64)
        raw_count = 0
        centered_cross = np.zeros((5, 5), dtype=np.float64)
        centered_count = 0
        parquet_cache = {}
        used_windows = 0
        for key in chosen:
            _, session_id, episode_index, window_index = parse_key(key)
            parquet_path = (
                root
                / f"task_{task_id}"
                / session_id
                / "data"
                / "data"
                / f"chunk-{episode_index // 1000:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            cache_key = str(parquet_path)
            if cache_key not in parquet_cache:
                frame = pd.read_parquet(parquet_path, columns=["timestamp", "observation.state"])
                timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
                states = np.stack(
                    [np.asarray(value, dtype=np.float64) for value in frame["observation.state"]]
                )
                order = np.argsort(timestamps)
                parquet_cache[cache_key] = (timestamps[order], states[order])
            timestamps, states = parquet_cache[cache_key]
            start = float(window_index) * float(args.stride_seconds)
            query = start + np.arange(args.frames_per_window, dtype=np.float64) / float(args.fps)
            indices = nearest_indices(query, timestamps)
            waist = states[indices, 85:90]
            waist = waist[np.isfinite(waist).all(axis=1)]
            if len(waist) < 2:
                continue
            raw_sum += waist.sum(axis=0)
            raw_count += len(waist)
            centered = waist - waist.mean(axis=0, keepdims=True)
            centered_cross += centered.T @ centered
            centered_count += len(centered)
            used_windows += 1
        if raw_count < 2 or centered_count < 2:
            raise RuntimeError(f"Insufficient waist samples for task {task_id}")
        task_raw_sums[task_id] = raw_sum
        task_raw_counts[task_id] = raw_count
        task_centered_covariances[task_id] = centered_cross / float(centered_count - 1)
        task_centered_counts[task_id] = centered_count
        task_windows[task_id] = used_windows

    task_ids = sorted(task_raw_sums)
    task_means = [task_raw_sums[task] / task_raw_counts[task] for task in task_ids]
    mean = np.mean(task_means, axis=0)
    covariance = np.mean([task_centered_covariances[task] for task in task_ids], axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = canonicalize_component_signs(eigenvectors[:, order].T[:2])
    explained = eigenvalues / max(float(eigenvalues.sum()), np.finfo(np.float64).eps)

    payload = {
        "version": "wholebody_cartesian_v1_waist_pca_encoder_train_task_balanced_v1",
        "created_from": str(split_path),
        "split_sha1": hashlib.sha1(split_path.read_bytes()).hexdigest(),
        "agibot_root": str(root),
        "tasks": task_ids,
        "seed": int(args.seed),
        "max_windows_per_task": int(args.max_windows_per_task),
        "frames_per_window": int(args.frames_per_window),
        "fps": float(args.fps),
        "stride_seconds": float(args.stride_seconds),
        "task_windows": task_windows,
        "task_sample_counts": task_raw_counts,
        "mean": mean.tolist(),
        "components": components.tolist(),
        "explained_variance_ratio": explained.tolist(),
        "explained_variance_ratio_first_two": float(explained[:2].sum()),
    }
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
