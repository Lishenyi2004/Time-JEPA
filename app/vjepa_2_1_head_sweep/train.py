"""Isolated V-JEPA 2.1 entry point for the EgoExo physical-head sweep.

Only this app process receives the experimental builder.  The production
``app.vjepa_2_1.train`` source remains unchanged, so already-running jobs are
not affected.
"""

from app.vjepa_2_1 import train as base_train
from src.time_jepa.physical_aux_head_sweep import (
    SUPPORTED_PHYSICAL_HEAD_TYPES,
    build_physical_head,
)


def main(args, resume_preempt=False):
    head_seed = int((args.get("meta", {}) or {}).get("seed", 0)) + 1009
    head_index = 0

    def build_sweep_head(**kwargs):
        nonlocal head_index
        # Head sizes consume different numbers of random draws.  Forking the
        # CPU RNG keeps the subsequent data-loader/augmentation/mask stream
        # paired across variants while retaining deterministic head init.
        with base_train.torch.random.fork_rng(devices=[]):
            base_train.torch.default_generator.manual_seed(
                head_seed + head_index
            )
            head = build_physical_head(**kwargs)
        head_index += 1
        return head

    base_train.SUPPORTED_PHYSICAL_HEAD_TYPES = SUPPORTED_PHYSICAL_HEAD_TYPES
    base_train.build_physical_head = build_sweep_head
    return base_train.main(args=args, resume_preempt=resume_preempt)
