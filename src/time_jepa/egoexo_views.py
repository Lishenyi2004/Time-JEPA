"""Lightweight EgoExo camera-role helpers (no torch/video dependencies)."""

import pathlib
import zlib

import numpy as np


def normalize_egoexo_video_views(video_view):
    if isinstance(video_view, str):
        values = video_view.split(",")
    elif isinstance(video_view, (list, tuple)):
        values = video_view
    else:
        values = [video_view]
    views = []
    for value in values:
        view = str(value).strip()
        if view and view not in views:
            views.append(view)
    if not views:
        raise ValueError("EgoExo requires at least one video view")
    return tuple(views)


def resolve_egoexo_video_candidate(root, take, requested_view):
    """Resolve a canonical role while retaining the take's physical camera name."""
    exo_view_paths = take.get("exo_view_paths") or {}
    exo_view_names = take.get("exo_view_names") or {}
    if requested_view in exo_view_paths:
        return {
            "video_path": exo_view_paths[requested_view],
            "video_view": requested_view,
            "video_role": requested_view,
            "physical_video_view": exo_view_names.get(requested_view, requested_view),
        }
    if requested_view == "best_exo":
        return {
            "video_path": take.get("best_exo_path"),
            "video_view": take.get("best_exo"),
            "video_role": "best_exo",
            "physical_video_view": take.get("best_exo"),
        }
    if requested_view in ("aria", "aria_rgb", "aria01_214-1"):
        return {
            "video_path": take.get("aria_rgb_path"),
            "video_view": "aria01_214-1",
            "video_role": "aria",
            "physical_video_view": "aria01_214-1",
        }
    video_path = (
        pathlib.Path(root)
        / "takes"
        / take["take_name"]
        / "frame_aligned_videos"
        / "downscaled"
        / "448"
        / f"{requested_view}.mp4"
    )
    return {
        "video_path": str(video_path),
        "video_view": requested_view,
        "video_role": requested_view,
        "physical_video_view": requested_view,
    }


def select_egoexo_video_candidate(row, random_view=True):
    """Return a shallow row copy with one candidate view selected."""
    candidates = row.get("video_candidates") or [
        {
            "video_path": row["video_path"],
            "video_view": row["video_view"],
            "video_role": row.get("video_role", row["video_view"]),
            "physical_video_view": row.get("physical_video_view", row["video_view"]),
        }
    ]
    if len(candidates) == 1:
        candidate_index = 0
    elif random_view:
        candidate_index = int(np.random.randint(len(candidates)))
    else:
        stable_key = str(row.get("window_key", row.get("take_uid", ""))).encode("utf-8")
        candidate_index = zlib.crc32(stable_key) % len(candidates)
    selected = dict(row)
    selected.update(candidates[candidate_index])
    selected["video_candidate_index"] = candidate_index
    return selected
