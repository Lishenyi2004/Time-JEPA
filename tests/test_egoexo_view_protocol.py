import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from src.time_jepa.egoexo_views import (
    normalize_egoexo_video_views,
    resolve_egoexo_video_candidate,
    select_egoexo_video_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "covla_probe_rjob"))
from probe_view_metrics import macro_average_view_metrics  # noqa: E402


SPLIT_SCRIPT = REPO_ROOT / "time-jepa" / "make_egoexo_splits.py"
SPEC = importlib.util.spec_from_file_location("make_egoexo_splits", SPLIT_SCRIPT)
SPLIT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLIT_MODULE)


def _candidate(role, physical_name):
    return {
        "video_path": f"/{physical_name}.mp4",
        "video_view": role,
        "video_role": role,
        "physical_video_view": physical_name,
    }


def test_split_maps_canonical_roles_to_one_complete_exo_rig(tmp_path):
    for camera in ("gp01", "gp02", "gp03", "gp04"):
        (tmp_path / f"{camera}.mp4").touch()

    prefix, names, paths = SPLIT_MODULE.resolve_exo_view_roles(
        tmp_path, ("view01", "view02", "view03", "view04")
    )

    assert prefix == "gp"
    assert names == {
        "view01": "gp01",
        "view02": "gp02",
        "view03": "gp03",
        "view04": "gp04",
    }
    assert all(path.is_file() for path in paths.values())


def test_split_rejects_an_incomplete_exo_rig(tmp_path):
    for camera in ("cam01", "cam02", "cam03"):
        (tmp_path / f"{camera}.mp4").touch()

    prefix, names, paths = SPLIT_MODULE.resolve_exo_view_roles(
        tmp_path, ("view01", "view02", "view03", "view04")
    )

    assert prefix is None
    assert names == {}
    assert paths == {}


def test_exo_view_ood_candidates_do_not_require_aria(tmp_path):
    take = {
        "take_uid": "take_uid_1",
        "take_name": "take_001",
        "participant_uid": "participant_1",
        "capture_uid": "capture_1",
        "duration_sec": 20.0,
        "validated": True,
    }
    (tmp_path / "takes.json").write_text(json.dumps([take]))
    pose_path = tmp_path / "annotations" / "ego_pose" / "train" / "body" / "automatic" / "take_uid_1.json"
    pose_path.parent.mkdir(parents=True)
    pose_path.write_text("{}")
    video_dir = tmp_path / "takes" / "take_001" / "frame_aligned_videos" / "downscaled" / "448"
    video_dir.mkdir(parents=True)
    for camera in ("cam01", "cam02", "cam03", "cam04"):
        (video_dir / f"{camera}.mp4").touch()

    rows, skipped = SPLIT_MODULE.candidate_takes(
        tmp_path,
        frames_per_clip=8,
        fps=5.0,
        stride_seconds=1.6,
        view_protocol="exo_view_ood",
        seen_views=("view01", "view02", "view03"),
        ood_views=("view04",),
    )

    assert len(rows) == 1
    assert rows[0]["aria_rgb_path"] is None
    assert rows[0]["exo_view_names"]["view04"] == "cam04"
    assert "missing_aria_rgb" not in skipped


def test_canonical_view_resolution_keeps_role_and_physical_camera_distinct():
    take = {
        "take_name": "take_001",
        "exo_view_paths": {"view04": "/data/gp04.mp4"},
        "exo_view_names": {"view04": "gp04"},
    }

    candidate = resolve_egoexo_video_candidate("/unused", take, "view04")

    assert candidate["video_view"] == "view04"
    assert candidate["physical_video_view"] == "gp04"
    assert candidate["video_path"] == "/data/gp04.mp4"


def test_train_sampling_is_uniform_without_multiplying_windows():
    row = {
        "window_key": "take/window_000001",
        "video_candidates": [
            _candidate("view01", "cam01"),
            _candidate("view02", "cam02"),
            _candidate("view03", "cam03"),
        ],
    }
    np.random.seed(239)
    counts = {view: 0 for view in normalize_egoexo_video_views("view01,view02,view03")}
    for _ in range(6000):
        selected = select_egoexo_video_candidate(row, random_view=True)
        counts[selected["video_view"]] += 1

    assert sum(counts.values()) == 6000
    assert all(abs(count - 2000) < 150 for count in counts.values())


def test_validation_view_assignment_is_stable_and_balanced():
    candidates = [
        _candidate("view01", "cam01"),
        _candidate("view02", "cam02"),
        _candidate("view03", "cam03"),
    ]
    first_pass = []
    second_pass = []
    for index in range(600):
        row = {"window_key": f"take/window_{index:06d}", "video_candidates": candidates}
        first_pass.append(select_egoexo_video_candidate(row, random_view=False)["video_view"])
        second_pass.append(select_egoexo_video_candidate(row, random_view=False)["video_view"])

    assert first_pass == second_pass
    assert all(abs(first_pass.count(view) - 200) < 35 for view in ("view01", "view02", "view03"))


def test_view_metrics_macro_includes_main_id_and_seen_views():
    macro = macro_average_view_metrics(
        {
            "view01": {"loss": 1.0, "r2": [0.1, 0.4]},
            "view02": {"loss": 2.0, "r2": [0.2, 0.5]},
            "view03": {"loss": 3.0, "r2": [0.3, 0.6]},
        }
    )

    assert macro["loss"] == 2.0
    np.testing.assert_allclose(macro["r2"], [0.2, 0.5])
