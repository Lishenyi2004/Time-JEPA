"""Reproduce the capture-date-disjoint CoVLA rainy-weather OOD protocol.

All clips from the eight rainy-heavy dates are removed from encoder/probe/ID
roles.  The main OOD test keeps clips labelled ``rainy`` in at least 8 of 10
uniformly sampled caption frames.  Source roles are also disjoint by complete
JST capture date.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


OOD_DATES = (
    "2022-07-13",
    "2022-07-14",
    "2022-09-07",
    "2022-09-18",
    "2022-10-05",
    "2022-10-06",
    "2022-10-13",
    "2022-12-22",
)

# These date assignments are frozen protocol data, not regenerated per run.
ROLE_DATES = {
    "encoder_train": (
        "2022-07-07", "2022-07-19", "2022-07-21", "2022-07-25",
        "2022-07-27", "2022-07-28", "2022-08-04", "2022-08-08",
        "2022-08-09", "2022-08-18", "2022-08-22", "2022-08-24",
        "2022-08-26", "2022-08-29", "2022-09-05", "2022-09-08",
        "2022-09-12", "2022-09-15", "2022-09-16", "2022-09-21",
        "2022-09-22", "2022-09-26", "2022-10-12", "2022-10-17",
        "2022-10-20", "2022-10-21", "2022-10-22", "2022-10-26",
        "2022-10-27", "2022-10-28", "2022-11-04", "2022-11-09",
        "2022-11-10", "2022-11-16", "2022-11-18", "2022-11-21",
        "2022-11-24", "2022-11-28", "2022-11-30", "2022-12-01",
        "2022-12-02", "2022-12-09", "2022-12-12", "2022-12-14",
        "2022-12-15", "2022-12-19", "2022-12-23", "2022-12-26",
    ),
    "probe_train": (
        "2022-07-20", "2022-07-29", "2022-08-17", "2022-08-31",
        "2022-09-06", "2022-09-14", "2022-10-24", "2022-11-07",
        "2022-11-25", "2022-12-07", "2022-12-08", "2022-12-20",
    ),
    "probe_val": (
        "2022-07-08", "2022-07-15", "2022-08-01", "2022-09-13",
        "2022-09-19", "2022-10-23", "2022-11-17", "2022-12-06",
    ),
    "test": (
        "2022-07-11", "2022-07-12", "2022-07-26", "2022-08-25",
        "2022-09-01", "2022-09-20", "2022-10-31", "2022-11-02",
        "2022-11-12", "2022-12-05",
    ),
}

JST = timezone(timedelta(hours=9), name="JST")
TIMESTAMP_RE = re.compile(rb'"timestamp"\s*:\s*([0-9]+)')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--covla-root",
        default="data/CoVLA-Dataset",
    )
    parser.add_argument("--output-dir", default="time-jepa/splits/covla_weather_ood")
    parser.add_argument("--weather-samples", type=int, default=10)
    parser.add_argument("--weather-frame-stride", type=int, default=60)
    parser.add_argument("--minimum-rainy-votes", type=int, default=8)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--scan-chunk-size", type=int, default=250)
    parser.add_argument("--scan-workers", type=int, default=1)
    return parser.parse_args()


def read_metadata(root):
    rows = []
    seen = set()
    with (root / "metadata.jsonl").open("r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            video_id = record.get("video_id")
            file_name = record.get("file_name")
            if not video_id or not file_name:
                raise RuntimeError(f"Invalid metadata row {line_number}: {record}")
            if video_id in seen:
                raise RuntimeError(f"Duplicate video_id: {video_id}")
            seen.add(video_id)
            rows.append((video_id, file_name))
    return rows


def scan_weather(path, sample_count, frame_stride):
    votes = Counter()
    rainy_confidences = []
    sampled_frames = []
    with path.open("rb") as handle:
        for line_index, line in enumerate(handle):
            if line_index % frame_stride:
                continue
            record = json.loads(line)
            if not record:
                continue
            frame_text, annotation = next(iter(record.items()))
            annotation = annotation or {}
            weather = str(annotation.get("weather", "unknown")).strip().lower()
            votes[weather] += 1
            if weather == "rainy":
                try:
                    rainy_confidences.append(float(annotation.get("weather_rate")))
                except (TypeError, ValueError):
                    pass
            sampled_frames.append(int(frame_text))
            if len(sampled_frames) == sample_count:
                break
    return {
        "sampled_frames": sampled_frames,
        "votes": dict(sorted(votes.items())),
        "rainy_votes": votes.get("rainy", 0),
        "rainy_confidence_mean": (
            sum(rainy_confidences) / len(rainy_confidences)
            if rainy_confidences
            else None
        ),
    }


def scan_chunk(task):
    root_text, rows, sample_count, frame_stride = task
    root = Path(root_text)
    ood_dates = set(OOD_DATES)
    results = []
    for video_id, file_name in rows:
        result = {"video_id": video_id}
        video_path = root / file_name
        state_path = root / "states" / f"{video_id}.jsonl"
        if not video_path.is_file():
            result["error"] = "missing_video"
            results.append(result)
            continue
        if not state_path.is_file():
            result["error"] = "missing_states"
            results.append(result)
            continue
        with state_path.open("rb") as handle:
            timestamp_match = TIMESTAMP_RE.search(handle.read(16384))
        timestamp_ms = int(timestamp_match.group(1)) if timestamp_match else 0
        if timestamp_ms <= 0:
            result["error"] = "missing_or_invalid_timestamp"
            results.append(result)
            continue
        capture_date = datetime.fromtimestamp(
            timestamp_ms / 1000.0,
            tz=timezone.utc,
        ).astimezone(JST).date().isoformat()
        result["capture_date"] = capture_date
        if capture_date in ood_dates:
            caption_path = root / "captions" / "captions" / f"{video_id}.jsonl"
            if not caption_path.is_file():
                result["weather_error"] = "missing_captions"
            else:
                try:
                    result["weather"] = scan_weather(
                        caption_path,
                        sample_count,
                        frame_stride,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    result["weather_error"] = f"caption_parse_error:{type(exc).__name__}"
        results.append(result)
    return results


def scan_dataset(root, rows, args):
    chunks = [
        rows[start : start + args.scan_chunk_size]
        for start in range(0, len(rows), args.scan_chunk_size)
    ]
    tasks = [
        (str(root), chunk, args.weather_samples, args.weather_frame_stride)
        for chunk in chunks
    ]
    with mp.get_context("fork").Pool(
        processes=args.scan_workers,
        maxtasksperchild=1,
    ) as pool:
        nested = pool.map(scan_chunk, tasks)
    return [item for chunk in nested for item in chunk]


def majority_weather(weather):
    votes = weather.get("votes", {})
    if not votes:
        return "unknown"
    priority = {"rainy": 0, "cloudy": 1, "sunny": 2, "unknown": 3}
    return min(votes, key=lambda name: (-votes[name], priority.get(name, 99), name))


def sha256_ids(video_ids):
    value = "".join(f"{video_id}\n" for video_id in video_ids).encode()
    return hashlib.sha256(value).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def assert_pairwise_disjoint(named_splits):
    names = list(named_splits)
    for index, left_name in enumerate(names):
        left = set(named_splits[left_name])
        if len(left) != len(named_splits[left_name]):
            raise RuntimeError(f"Duplicate ids inside {left_name}")
        for right_name in names[index + 1 :]:
            overlap = left & set(named_splits[right_name])
            if overlap:
                raise RuntimeError(f"Overlap {left_name}/{right_name}: {sorted(overlap)[:5]}")


def main():
    args = parse_args()
    if args.scan_chunk_size <= 0 or args.scan_workers <= 0:
        raise ValueError("scan chunk size and worker count must be positive")
    if args.weather_samples <= 0 or args.weather_frame_stride <= 0:
        raise ValueError("weather sample count and frame stride must be positive")
    if not 1 <= args.minimum_rainy_votes <= args.weather_samples:
        raise ValueError("minimum rainy votes must be in [1, weather_samples]")
    root = Path(args.covla_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = read_metadata(root)
    results = scan_dataset(root, rows, args)
    by_id = {result["video_id"]: result for result in results}
    if len(by_id) != len(rows):
        raise RuntimeError("scanner did not return exactly one result per metadata row")

    groups = defaultdict(list)
    quarantine_reasons = defaultdict(list)
    for video_id, _ in rows:
        result = by_id[video_id]
        if result.get("error"):
            quarantine_reasons[result["error"]].append(video_id)
        else:
            groups[result["capture_date"]].append(video_id)
    for video_ids in groups.values():
        video_ids.sort()

    frozen_dates = set(OOD_DATES)
    for dates in ROLE_DATES.values():
        frozen_dates.update(dates)
    unknown_dates = sorted(set(groups) - frozen_dates)
    missing_dates = sorted(frozen_dates - set(groups))
    if unknown_dates or missing_dates:
        raise RuntimeError(
            f"Dataset date inventory differs from frozen protocol; "
            f"unknown={unknown_dates}, missing={missing_dates}"
        )

    source_splits = {
        role: sorted(video_id for date in dates for video_id in groups[date])
        for role, dates in ROLE_DATES.items()
    }
    day_holdout = sorted(video_id for date in OOD_DATES for video_id in groups[date])
    stable_rain = []
    high_confidence_rain = []
    majority_counts = Counter()
    vote_histogram = Counter()
    weather_errors = defaultdict(list)
    weather_audit = {}
    for video_id in day_holdout:
        result = by_id[video_id]
        if result.get("weather_error"):
            weather_errors[result["weather_error"]].append(video_id)
            continue
        weather = result.get("weather") or {}
        weather_audit[video_id] = weather
        majority_counts[majority_weather(weather)] += 1
        rainy_votes = int(weather.get("rainy_votes", 0))
        vote_histogram[rainy_votes] += 1
        if rainy_votes >= args.minimum_rainy_votes:
            stable_rain.append(video_id)
            confidence = weather.get("rainy_confidence_mean")
            if confidence is not None and confidence >= args.high_confidence_threshold:
                high_confidence_rain.append(video_id)

    quarantine = sorted(
        video_id
        for video_ids in quarantine_reasons.values()
        for video_id in video_ids
    )
    partition = {
        **source_splits,
        "weather_ood_day_holdout": day_holdout,
        "quarantine": quarantine,
    }
    assert_pairwise_disjoint(partition)
    metadata_ids = {video_id for video_id, _ in rows}
    partition_ids = {video_id for ids in partition.values() for video_id in ids}
    if partition_ids != metadata_ids:
        raise RuntimeError("Frozen split partition does not cover metadata exactly")

    payloads = {
        **source_splits,
        "weather_ood_test": stable_rain,
        "weather_ood_high_conf_test": high_confidence_rain,
        "weather_ood_day_holdout": day_holdout,
        "quarantine": quarantine,
    }
    files = {}
    for name, video_ids in payloads.items():
        path = output_dir / f"covla_{name}.json"
        write_json(path, video_ids)
        files[name] = str(path)

    manifest = {
        "protocol": "covla_weather_ood_v1",
        "covla_root": str(root),
        "timezone": "Asia/Tokyo (UTC+09:00)",
        "definition": {
            "group_unit": "complete JST capture date",
            "source_role_ratios": {
                "encoder_train": 0.70,
                "probe_train": 0.15,
                "probe_val": 0.05,
                "test": 0.10,
            },
            "ood_dates": list(OOD_DATES),
            "ood_day_policy": "all clips on OOD dates are excluded from every source role",
            "main_ood_test": (
                f"rainy in at least {args.minimum_rainy_votes}/{args.weather_samples} "
                "uniform caption samples"
            ),
            "weather_sample_frames": [
                index * args.weather_frame_stride
                for index in range(args.weather_samples)
            ],
            "high_confidence_threshold": args.high_confidence_threshold,
        },
        "counts": {
            **{name: len(ids) for name, ids in payloads.items()},
            "metadata_total": len(rows),
            "source_total": sum(len(ids) for ids in source_splits.values()),
        },
        "dates": {**{key: list(value) for key, value in ROLE_DATES.items()},
                  "weather_ood_day_holdout": list(OOD_DATES)},
        "date_counts": {date: len(ids) for date, ids in sorted(groups.items())},
        "ood_weather_audit": {
            "day_holdout_majority_weather": dict(sorted(majority_counts.items())),
            "rainy_vote_histogram": {
                str(votes): count for votes, count in sorted(vote_histogram.items())
            },
            "weather_errors": {
                reason: sorted(ids) for reason, ids in sorted(weather_errors.items())
            },
            "per_video": weather_audit,
        },
        "quarantine_reasons": {
            reason: sorted(ids) for reason, ids in sorted(quarantine_reasons.items())
        },
        "checks": {
            "partition_is_pairwise_disjoint": True,
            "partition_covers_all_metadata": True,
            "source_roles_are_capture_date_disjoint": True,
            "main_ood_is_subset_of_day_holdout": set(stable_rain) <= set(day_holdout),
        },
        "sha256": {name: sha256_ids(ids) for name, ids in payloads.items()},
        "files": files,
    }
    write_json(output_dir / "covla_split_manifest_weather_ood.json", manifest)
    summary = dict(manifest)
    summary["ood_weather_audit"] = {
        key: value for key, value in manifest["ood_weather_audit"].items()
        if key != "per_video"
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
