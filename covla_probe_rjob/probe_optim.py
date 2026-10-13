"""Optimizer and warmup-cosine LR / cosine WD schedules."""

import math

import torch


class WarmupCosineLRSchedule:
    def __init__(self, optimizer, total_steps):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.step_num = 0

    def step(self):
        self.step_num += 1
        for group in self.optimizer.param_groups:
            ref_lr = group["mc_ref_lr"]
            final_lr = group["mc_final_lr"]
            start_lr = group["mc_start_lr"]
            warmup_steps = group["mc_warmup_steps"]
            if self.step_num < warmup_steps:
                progress = float(self.step_num) / float(max(1, warmup_steps))
                lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self.step_num - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
                lr = final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = max(final_lr, lr) if final_lr <= ref_lr else min(final_lr, lr)
            group["lr"] = lr


class CosineWDSchedule:
    def __init__(self, optimizer, total_steps):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.step_num = 0

    def step(self):
        self.step_num += 1
        progress = float(self.step_num) / float(self.total_steps)
        for group in self.optimizer.param_groups:
            ref_wd = group["mc_ref_wd"]
            final_wd = group["mc_final_wd"]
            wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            wd = max(final_wd, wd) if final_wd <= ref_wd else min(final_wd, wd)
            group["weight_decay"] = wd


def build_optimizer_and_schedulers(args, probe, iterations_per_epoch):
    warmup_steps = int(args.warmup_epochs * iterations_per_epoch)
    start_lr = args.start_lr if args.start_lr is not None else args.lr
    final_weight_decay = args.final_weight_decay if args.final_weight_decay is not None else args.weight_decay
    param_groups = [
        {
            "params": [p for p in probe.parameters() if p.requires_grad],
            "lr": start_lr,
            "weight_decay": args.weight_decay,
            "mc_warmup_steps": warmup_steps,
            "mc_start_lr": start_lr,
            "mc_ref_lr": args.lr,
            "mc_final_lr": args.final_lr,
            "mc_ref_wd": args.weight_decay,
            "mc_final_wd": final_weight_decay,
        }
    ]
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = max(1, args.epochs * iterations_per_epoch)
    if args.lr_schedule == "cosine":
        lr_scheduler = WarmupCosineLRSchedule(optimizer, total_steps=total_steps)
        wd_scheduler = CosineWDSchedule(optimizer, total_steps=total_steps)
    else:
        lr_scheduler = None
        wd_scheduler = None
    return optimizer, lr_scheduler, wd_scheduler
