import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_COUNTS = {"encoder_train": 7000, "probe_train": 1500, "probe_val": 500, "test": 1000}
DEFAULT_RATIOS = {"encoder_train": 0.70, "probe_train": 0.15, "probe_val": 0.05, "test": 0.10}
DEFAULT_SEEN_VIEWS = ("view01", "view02", "view03")
DEFAULT_OOD_VIEWS = ("view04",)
EXO_RIG_PREFIXES = ("cam", "gp")


class UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def load_takes(root):
    with (root / "takes.json").open("r") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    for value in payload.values():
        if isinstance(value, list):
            return value
    raise RuntimeError("Could not find take list in takes.json")


def pose_paths(root):
    mapping = {}
    for split in ("train", "val"):
        directory = root / "annotations" / "ego_pose" / split / "body" / "automatic"
        for path in directory.glob("*.json"):
            mapping[path.stem] = path
    return mapping


def window_count(duration, frames_per_clip, fps, stride_seconds):
    span = (int(frames_per_clip) - 1) / float(fps)
    max_start = float(duration) - span
    if max_start < 0:
        return 0
    return int(math.floor(max_start / float(stride_seconds) + 1e-9)) + 1


def parse_view_roles(value):
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    roles = []
    for raw_value in values:
        role = str(raw_value).strip().lower()
        if not role:
            continue
        match = re.fullmatch(r"view0*([1-9][0-9]*)", role)
        if match is None:
            raise ValueError(f"Expected canonical exo view role such as view01, got {raw_value!r}")
        normalized = f"view{int(match.group(1)):02d}"
        if normalized not in roles:
            roles.append(normalized)
    if not roles:
        raise ValueError("At least one canonical exo view role is required")
    return tuple(roles)


def resolve_exo_view_roles(video_dir, required_views):
    """Map canonical view01/... roles to a take's cam01/... or gp01/... files."""
    for prefix in EXO_RIG_PREFIXES:
        names = {
            role: f"{prefix}{int(role[len('view'):]):02d}"
            for role in required_views
        }
        paths = {role: video_dir / f"{name}.mp4" for role, name in names.items()}
        if all(path.is_file() for path in paths.values()):
            return prefix, names, paths
    return None, {}, {}


def candidate_takes(
    root,
    frames_per_clip,
    fps,
    stride_seconds,
    view_protocol="bestexo_aria",
    seen_views=DEFAULT_SEEN_VIEWS,
    ood_views=DEFAULT_OOD_VIEWS,
):
    seen_views = parse_view_roles(seen_views)
    ood_views = parse_view_roles(ood_views)
    required_views = tuple(dict.fromkeys((*seen_views, *ood_views)))
    if set(seen_views) & set(ood_views):
        raise ValueError(f"Seen/OOD exo view roles overlap: {sorted(set(seen_views) & set(ood_views))}")
    poses = pose_paths(root)
    candidates = []
    skipped = defaultdict(int)
    for take in load_takes(root):
        uid = str(take.get("take_uid", ""))
        pose_path = poses.get(uid)
        if pose_path is None:
            skipped["missing_pose"] += 1
            continue
        if take.get("validated") is False:
            skipped["unvalidated"] += 1
            continue
        duration = float(take.get("duration_sec") or 0.0)
        count = window_count(duration, frames_per_clip, fps, stride_seconds)
        if count <= 0:
            skipped["too_short"] += 1
            continue
        video_dir = root / "takes" / take["take_name"] / "frame_aligned_videos" / "downscaled" / "448"
        best_exo = take.get("best_exo")
        best_path = video_dir / f"{best_exo}.mp4" if best_exo else None
        aria_path = video_dir / "aria01_214-1.mp4"
        exo_rig_prefix, exo_view_names, exo_view_paths = resolve_exo_view_roles(video_dir, required_views)
        if view_protocol == "bestexo_aria":
            if best_path is None or not best_path.is_file():
                skipped["missing_best_exo"] += 1
                continue
            if not aria_path.is_file():
                skipped["missing_aria_rgb"] += 1
                continue
        elif view_protocol == "exo_view_ood":
            if exo_rig_prefix is None:
                skipped["missing_required_exo_views"] += 1
                continue
        else:
            raise ValueError(f"Unknown EgoExo view protocol: {view_protocol}")
        row = {
            "take_uid": uid,
            "take_name": take["take_name"],
            "participant_uid": str(take.get("participant_uid", "")),
            "capture_uid": str(take.get("capture_uid", "")),
            "task_id": str(take.get("task_id", "")),
            "physical_setting_uid": str(take.get("physical_setting_uid", "")),
            "duration_sec": duration,
            "num_windows": count,
            "best_exo": best_exo,
            "best_exo_path": str(best_path) if best_path is not None and best_path.is_file() else None,
            "aria_rgb_path": str(aria_path) if aria_path.is_file() else None,
            "pose_path": str(pose_path),
        }
        if exo_rig_prefix is not None:
            row.update({
                "exo_rig_prefix": exo_rig_prefix,
                "exo_view_names": exo_view_names,
                "exo_view_paths": {role: str(path) for role, path in exo_view_paths.items()},
            })
        candidates.append(row)
    return candidates, dict(skipped)


def connected_components(rows):
    uf = UnionFind([row["take_uid"] for row in rows])
    groups = {}
    for field in ("participant_uid", "capture_uid"):
        first = {}
        for row in rows:
            value = row[field]
            if not value:
                continue
            if value in first:
                uf.union(row["take_uid"], first[value])
            else:
                first[value] = row["take_uid"]
    components = defaultdict(list)
    for row in rows:
        components[uf.find(row["take_uid"])].append(row)
    return list(components.values())


def assign_components(components, ratios, seed):
    rng = random.Random(seed)
    components = list(components)
    rng.shuffle(components)
    components.sort(key=lambda group: sum(row["num_windows"] for row in group), reverse=True)
    total = sum(row["num_windows"] for group in components for row in group)
    targets = {name: total * ratio for name, ratio in ratios.items()}
    assigned = {name: [] for name in ratios}
    counts = {name: 0 for name in ratios}
    for group in components:
        size = sum(row["num_windows"] for row in group)
        name = max(ratios, key=lambda key: targets[key] - counts[key])
        assigned[name].extend(group)
        counts[name] += size
    return assigned


def sample_keys(rows, count, stride_seconds, seed, split_name):
    keys = [f"{row['take_uid']}/window_{index:06d}" for row in rows for index in range(row["num_windows"])]
    if len(keys) < count:
        raise RuntimeError(f"EgoExo split {split_name} has {len(keys)} windows, requested {count}")
    rng = random.Random(f"{seed}:{split_name}:windows")
    rng.shuffle(keys)
    return sorted(keys[:count]), len(keys)


def assert_disjoint(assigned, split_keys):
    names = list(assigned)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for field in ("take_uid", "participant_uid", "capture_uid"):
                left_values = {row[field] for row in assigned[left] if row[field]}
                right_values = {row[field] for row in assigned[right] if row[field]}
                assert not left_values & right_values, f"{field} leakage: {left} vs {right}"
            assert not set(split_keys[left]) & set(split_keys[right]), f"window leakage: {left} vs {right}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/EgoExo4D_minimal")
    parser.add_argument("--output-dir", default="time-jepa/splits")
    parser.add_argument("--prefix", default="egoexo_bestexo")
    parser.add_argument("--frames-per-clip", type=int, default=64)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--slice-stride-seconds", type=float, default=12.8)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument(
        "--view-protocol",
        choices=("bestexo_aria", "exo_view_ood"),
        default="bestexo_aria",
        help=(
            "bestexo_aria preserves the original best-exo/Aria protocol. "
            "exo_view_ood requires a single cam or GoPro rig with all requested canonical views."
        ),
    )
    parser.add_argument(
        "--seen-views",
        default=",".join(DEFAULT_SEEN_VIEWS),
        help="Canonical exo roles used during training, for example view01,view02,view03.",
    )
    parser.add_argument(
        "--ood-views",
        default=",".join(DEFAULT_OOD_VIEWS),
        help="Canonical held-out exo roles, for example view04.",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    seen_views = parse_view_roles(args.seen_views)
    ood_views = parse_view_roles(args.ood_views)
    rows, skipped = candidate_takes(
        root,
        args.frames_per_clip,
        args.fps,
        args.slice_stride_seconds,
        view_protocol=args.view_protocol,
        seen_views=seen_views,
        ood_views=ood_views,
    )
    assigned = assign_components(connected_components(rows), DEFAULT_RATIOS, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_keys = {}
    capacity = {}
    files = {}
    for name, count in DEFAULT_COUNTS.items():
        keys, split_capacity = sample_keys(assigned[name], count, args.slice_stride_seconds, args.seed, name)
        path = output_dir / f"{args.prefix}_{name}.json"
        path.write_text(json.dumps(keys, indent=2))
        split_keys[name] = keys
        capacity[name] = split_capacity
        files[name] = str(path.resolve())
    assert_disjoint(assigned, split_keys)
    take_index = {row["take_uid"]: row for row in rows}
    manifest = {
        "root": str(root),
        "seed": args.seed,
        "key_format": "<take_uid>/window_<window_idx:06d>",
        "frames_per_clip": args.frames_per_clip,
        "fps": args.fps,
        "slice_stride_seconds": args.slice_stride_seconds,
        "view_protocol": args.view_protocol,
        "seen_views": list(seen_views),
        "ood_views": list(ood_views),
        "ratios": DEFAULT_RATIOS,
        "requested_counts": DEFAULT_COUNTS,
        "counts": {name: len(keys) for name, keys in split_keys.items()},
        "capacity_windows": capacity,
        "split_summary": {
            name: {
                "takes": len(group),
                "participants": len({row["participant_uid"] for row in group}),
                "captures": len({row["capture_uid"] for row in group}),
                "hours": sum(row["duration_sec"] for row in group) / 3600.0,
            }
            for name, group in assigned.items()
        },
        "skipped": skipped,
        "take_index": take_index,
        "files": files,
    }
    manifest_path = output_dir / f"{args.prefix}_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key != "take_index"}, indent=2))


if __name__ == "__main__":
    main()
