"""Compute train-only 40-D action normalization for the AgiBot PSG adapter."""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import nearest_indices
from .protocol import DEFAULT_ACTION_DIM, DEFAULT_MICROCLIP_FRAMES, AgiBotPSGProtocol, resolve_record


@lru_cache(maxsize=64)
def _load_action_series(parquet_path: str) -> tuple[np.ndarray, np.ndarray]:
    dataframe = pd.read_parquet(
        parquet_path,
        columns=["timestamp", "frame_index", "action"],
    )
    timestamps = dataframe["timestamp"].to_numpy(dtype=np.float64)
    frame_indices = dataframe["frame_index"].to_numpy(dtype=np.int64)
    actions = np.stack(
        [np.asarray(value, dtype=np.float64) for value in dataframe["action"]],
        axis=0,
    )
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    frame_indices = frame_indices[order]
    actions = actions[order]
    if actions.shape[1:] != (DEFAULT_ACTION_DIM,):
        raise ValueError(f"Expected 40-D action in {parquet_path}, got {actions.shape}")
    if not np.array_equal(np.diff(frame_indices), np.ones(len(frame_indices) - 1, dtype=np.int64)):
        raise ValueError(f"Non-contiguous frame_index in {parquet_path}")
    return timestamps, actions


def compute_action_stats(
    protocol: AgiBotPSGProtocol,
    *,
    max_windows: int | None = None,
) -> dict[str, object]:
    """Accumulate exactly the raw commands consumed by grid-packed training clips."""

    sample_keys = protocol.sample_keys
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive")
        sample_keys = sample_keys[: int(max_windows)]

    sums = np.zeros(DEFAULT_ACTION_DIM, dtype=np.float64)
    squared_sums = np.zeros(DEFAULT_ACTION_DIM, dtype=np.float64)
    count = 0
    offsets = tuple(range(0, protocol.window_frames, DEFAULT_MICROCLIP_FRAMES))
    first_view = protocol.video_keys[0]
    for index, sample_key in enumerate(sample_keys, 1):
        record = resolve_record(protocol, sample_key=sample_key, video_key=first_view)
        timestamps, actions = _load_action_series(str(record.parquet_path))
        sample_times = record.start_time + np.arange(
            protocol.window_frames,
            dtype=np.float64,
        ) / protocol.model_fps
        rows = nearest_indices(
            sample_times,
            timestamps,
            max_delta=protocol.max_time_delta,
        )
        for offset in offsets:
            for step in range(DEFAULT_MICROCLIP_FRAMES - 1):
                start = int(rows[offset + step])
                stop = int(rows[offset + step + 1])
                block = actions[start:stop]
                expected = protocol.action_steps_per_model_step
                if block.shape != (expected, DEFAULT_ACTION_DIM):
                    raise ValueError(
                        f"Bad action block for {sample_key}: {block.shape}, "
                        f"expected ({expected}, {DEFAULT_ACTION_DIM})"
                    )
                sums += block.sum(axis=0)
                squared_sums += np.square(block).sum(axis=0)
                count += len(block)
        if index % 100 == 0 or index == len(sample_keys):
            print(f"action stats: {index}/{len(sample_keys)} physical windows", flush=True)

    if count < 2:
        raise RuntimeError(f"Insufficient action rows for statistics: count={count}")
    mean = sums / count
    variance = (squared_sums - np.square(sums) / count) / (count - 1)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.maximum(std, 1.0e-6)
    return {
        "meta": {
            "domain": "agibot",
            "kind": "psg_raw_action_40d",
            "params_path": str(protocol.params_path),
            "sample_seed": protocol.sample_seed,
            "num_physical_windows": len(sample_keys),
            "num_views_counted": 1,
            "raw_action_rows": count,
            "raw_fps": protocol.raw_fps,
            "model_fps": protocol.model_fps,
            "microclip_frames": DEFAULT_MICROCLIP_FRAMES,
            "microclips_per_window": protocol.microclips_per_window,
            "unused_fourth_action": "zero_padded_after_normalization",
        },
        "mean": mean.tolist(),
        "std": std.tolist(),
        "counts": [count] * DEFAULT_ACTION_DIM,
    }


def _atomic_json_dump(payload: dict[str, object], output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-windows", type=int)
    arguments = parser.parse_args()
    protocol = AgiBotPSGProtocol.from_params(arguments.params)
    payload = compute_action_stats(protocol, max_windows=arguments.max_windows)
    _atomic_json_dump(payload, Path(arguments.output))
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
