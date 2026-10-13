#!/usr/bin/env python3
"""Transcode AgiBot AV1 videos to H.264 for fast decode during training.

AgiBot ships videos as AV1 (640x400 / 640x528, 30fps). AV1 software decode is
very slow on CPU and is the dominant bottleneck for Time-JEPA / probe training
(the GPU spends most of its time waiting on the decode workers). This script
transcodes each AV1 mp4 into an H.264 mp4 that decord/ffmpeg can decode 5-10x
faster, written to a sibling ``videos_h264/`` directory that mirrors the
original ``videos/`` layout. The original AV1 files are never touched.

Layout (per task session dir):

    task_<id>/<session>/data/videos/chunk-000/<video_key>/episode_*.mp4      (AV1, original)
    task_<id>/<session>/data/videos_h264/chunk-000/<video_key>/episode_*.mp4 (H.264, produced)

The readers (src/time_jepa/agibot_time_dataset.py, covla_probe_rjob/probe_datasets.py)
prefer ``videos_h264`` when AGIBOT_VIDEO_DIRNAME=videos_h264 is set, and fall
back to the original ``videos`` file if a transcoded one is missing, so a
partial transcode is safe to train on (only the missing ones stay slow).

Usage:

    # transcode all default tasks with 16 parallel ffmpeg workers
    python3 time-jepa/transcode_agibot_h264.py --workers 16

    # only some tasks / views, dry-run first to see the plan
    python3 time-jepa/transcode_agibot_h264.py --tasks 3641,3777 \
        --video-keys observation.images.top_head,observation.images.hand_left \
        --dry-run

    # include the fisheye views for the new OOD protocol
    python3 time-jepa/transcode_agibot_h264.py \
        --video-keys observation.images.head_left_fisheye,observation.images.head_right_fisheye

Re-running is safe: already-transcoded outputs are skipped unless --overwrite.
Interrupted/partial output files are detected via a .tmp write + atomic rename,
so a killed job never leaves a corrupt mp4 in place.
"""

import argparse
import concurrent.futures as cf
import os
import pathlib
import shutil
import subprocess
import sys

DEFAULT_ROOT = "data/agibot"
DEFAULT_TASKS = [3400, 3401, 3402, 3404, 3405, 3477, 3641, 3705, 3777, 4053]
DEFAULT_VIDEO_KEYS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]
SRC_DIRNAME = "videos"
DST_DIRNAME = "videos_h264"


def discover_task_data_dirs(root, task_id):
    """Return list of (session_id, data_dir) for a task, matching the readers."""
    task_root = pathlib.Path(root) / f"task_{task_id}"
    old_style = task_root / "data"
    if (old_style / "meta" / "info.json").exists():
        return [(None, old_style)]
    data_dirs = []
    for data_dir in sorted(task_root.glob("*/data")):
        if (data_dir / "meta" / "info.json").exists():
            data_dirs.append((data_dir.parent.name, data_dir))
    return data_dirs


def find_jobs(root, tasks, video_keys):
    """Yield (src_path, dst_path) pairs for every AV1 mp4 to transcode."""
    jobs = []
    for task_id in tasks:
        for _session_id, data_dir in discover_task_data_dirs(root, task_id):
            src_videos = data_dir / SRC_DIRNAME
            if not src_videos.is_dir():
                continue
            for video_key in video_keys:
                # videos/chunk-000/<video_key>/episode_*.mp4
                for src in sorted(src_videos.glob(f"chunk-*/{video_key}/episode_*.mp4")):
                    rel = src.relative_to(src_videos)
                    dst = data_dir / DST_DIRNAME / rel
                    jobs.append((src, dst))
    return jobs


def transcode_one(job, crf, preset, overwrite, ffmpeg_binary):
    src, dst = job
    if dst.exists() and not overwrite:
        return ("skip", src, dst, "")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    # -map 0:v:0 keeps only the video stream; robot state lives in parquet, not
    # in the mp4, so audio/data streams are irrelevant. yuv420p + faststart make
    # the output maximally decode-friendly for decord.
    cmd = [
        ffmpeg_binary,
        "-y",
        "-v", "error",
        "-i", str(src),
        "-map", "0:v:0",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            if tmp.exists():
                tmp.unlink()
            err = proc.stderr.decode("utf-8", errors="replace").strip()[-500:]
            return ("fail", src, dst, err)
        os.replace(tmp, dst)
        return ("ok", src, dst, "")
    except Exception as exc:  # pragma: no cover - defensive
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return ("fail", src, dst, f"{type(exc).__name__}: {exc}")


def parse_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="AgiBot dataset root")
    parser.add_argument("--tasks", default=",".join(str(t) for t in DEFAULT_TASKS),
                        help="comma-separated task ids")
    parser.add_argument("--video-keys", default=",".join(DEFAULT_VIDEO_KEYS),
                        help="comma-separated observation.images.* views to transcode")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2),
                        help="parallel ffmpeg processes (default: cpu_count-2)")
    parser.add_argument("--crf", type=int, default=20,
                        help="H.264 CRF quality (lower=better/bigger; 18-23 is visually ~lossless)")
    parser.add_argument("--preset", default="veryfast",
                        help="x264 speed/size preset (ultrafast..veryslow); veryfast is a good decode-speed tradeoff")
    parser.add_argument("--overwrite", action="store_true", help="re-transcode even if output exists")
    parser.add_argument("--dry-run", action="store_true", help="list the plan and exit")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_BINARY", "ffmpeg"))
    args = parser.parse_args()

    if shutil.which(args.ffmpeg) is None:
        print(f"ERROR: ffmpeg binary not found: {args.ffmpeg}", file=sys.stderr)
        return 2

    tasks = parse_list(args.tasks)
    video_keys = parse_list(args.video_keys)
    jobs = find_jobs(args.root, tasks, video_keys)
    if not jobs:
        print(f"No AV1 mp4s found under {args.root} for tasks={tasks} views={video_keys}", file=sys.stderr)
        return 1

    todo = [j for j in jobs if args.overwrite or not j[1].exists()]
    already = len(jobs) - len(todo)
    print(f"Found {len(jobs)} videos ({already} already transcoded, {len(todo)} to do) "
          f"across {len(tasks)} tasks x {len(video_keys)} views")
    print(f"src '{SRC_DIRNAME}/' -> dst '{DST_DIRNAME}/'  crf={args.crf} preset={args.preset} workers={args.workers}")

    if args.dry_run:
        for src, dst in todo[:10]:
            print(f"  {src}\n    -> {dst}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    ok = skip = fail = 0
    failures = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(transcode_one, job, args.crf, args.preset, args.overwrite, args.ffmpeg)
            for job in todo
        ]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            status, src, _dst, err = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                if len(failures) < 20:
                    failures.append((src, err))
            if i % 50 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"Done. ok={ok} skip={skip} fail={fail}")
    if failures:
        print("First failures:", file=sys.stderr)
        for src, err in failures:
            print(f"  {src}\n    {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
