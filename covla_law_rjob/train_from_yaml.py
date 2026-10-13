#!/usr/bin/env python3
"""Law runner: keep generated split files separate across distributed ranks.

The shared runner's temporary metadata filename contains only the PID. PIDs
can repeat in different containers on a shared filesystem; a directory per
global rank avoids the cross-node rename race without editing that runner.
"""

import os
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from covla_jepa_rjob import train_from_yaml as shared_runner
import torch
import torch.distributed as dist


def main():
    original = shared_runner.apply_covla_train_split
    original_init = shared_runner.init_distributed

    def init_distributed(*args, **kwargs):
        world, rank = original_init(*args, **kwargs)
        probe = torch.ones(4, device=torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(probe)
        if not torch.equal(probe, torch.full_like(probe, world)):
            raise RuntimeError("Distributed CUDA communication preflight failed")
        logging.getLogger(__name__).info(
            "LAW_COMM_PREFLIGHT passed rank=%s world_size=%s NCCL_NET=%s NCCL_IB_DISABLE=%s",
            rank, world, os.environ.get("NCCL_NET"), os.environ.get("NCCL_IB_DISABLE"),
        )
        return world, rank

    def apply_split(params, split_path, filtered_metadata_dir=None):
        rank = int(os.environ.get("RANK", "0"))
        directory = Path(filtered_metadata_dir or Path(params["folder"]) / "filtered_metadata")
        directory = directory / f"rank_{rank:03d}"
        return original(params, split_path, filtered_metadata_dir=str(directory))

    shared_runner.apply_covla_train_split = apply_split
    shared_runner.init_distributed = init_distributed
    try:
        shared_runner.main()
    finally:
        shared_runner.apply_covla_train_split = original
        shared_runner.init_distributed = original_init


if __name__ == "__main__":
    main()
