"""Distributed, seeding, memory, and checkpoint-IO helpers."""

import gc
import os
import resource
from pathlib import Path

import numpy as np
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            return None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if args.device == "cpu" or not torch.cuda.is_available():
            backend = "gloo"
            device = torch.device("cpu")
        else:
            backend = "nccl"
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = torch.device(args.device)

    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
        "device": device,
    }


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def print_main(is_main, *args, **kwargs):
    if is_main:
        print(*args, **kwargs)


def reduce_sum(tensor, distributed):
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def build_summary_writer(args, output_dir, is_main):
    if not is_main or args.no_tensorboard:
        return None
    if SummaryWriter is None:
        print("tensorboard: unavailable; install tensorboard to enable event logging")
        return None
    log_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"tensorboard_log_dir={log_dir}")
    return SummaryWriter(log_dir=str(log_dir))


def release_epoch_memory(device, clear_state_cache=False):
    if clear_state_cache:
        from probe_targets import load_state_series, load_kitti_oxts_series

        load_state_series.cache_clear()
        load_kitti_oxts_series.cache_clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def memory_report(device):
    report = {"cpu_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    if device.type == "cuda":
        report.update(
            {
                "cuda_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "cuda_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
                "cuda_max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            }
        )
    return {k: round(v, 3) for k, v in report.items()}


def torch_load(path, map_location="cpu"):
    # Encoder provenance is stored alongside probe weights.  In recent PyTorch
    # releases ``torch.__version__`` is a ``TorchVersion`` (a ``str`` subclass),
    # so older checkpoints can legitimately contain that class.  Keep the
    # restricted weights-only loader and allowlist only this known-safe value.
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    try:
        from torch.torch_version import TorchVersion
    except ImportError:  # pragma: no cover - for older PyTorch releases
        TorchVersion = None

    try:
        if safe_globals is not None and TorchVersion is not None:
            with safe_globals([TorchVersion]):
                return torch.load(path, weights_only=True, map_location=map_location)
        return torch.load(path, weights_only=True, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)
