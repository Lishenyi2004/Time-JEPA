"""Isolated Time-JEPA + longitudinal law app using the production trainer.

The loader adapter carries the additional law metadata to the physical-loss
call in the *same* main-process batch. The original trainer still owns every
forward, backward, optimizer/EMA update, schedule and checkpoint boundary.
No production source or model architecture is changed.
"""

import copy
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path

import torch

from app.vjepa_2_1 import train as base_train
from src.time_jepa import covla_time_dataset
from src.time_jepa.covla_law import PROTOCOL, longitudinal_law_loss, sha256_file
from src.time_jepa.covla_law_dataset import CoVLALawDataset
from src.time_jepa.physical_targets import normalize_target_fields

ROOT = Path(__file__).resolve().parents[2]
BASE_TRAIN_SHA256 = "64b56164721635d3c14de1630ec8e2e8fc2a3117f9161dcc3d13677ddbaa63a8"
logger = logging.getLogger(__name__)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_config(args):
    cfg, data = args["time_jepa"], args["data"]
    law = args["explicit_physics_law"]
    if sha256_file(ROOT / "app/vjepa_2_1/train.py") != BASE_TRAIN_SHA256:
        raise RuntimeError("Production trainer changed: re-audit the isolated law adapter before use")
    if not cfg.get("enabled") or cfg.get("mode", "single") not in ("single", ""):
        raise ValueError("The law app supports single-domain CoVLA Time-JEPA")
    if normalize_target_fields(cfg["targets"]) != ["vEgo", "aEgo", "angular_velocities_calib[2]"]:
        raise ValueError("Keep the original three CoVLA state targets")
    if cfg["head_type"] != "conv1x1" or cfg["feature_mode"] != "final" or not cfg["disable_horizontal_flip"]:
        raise ValueError("Keep the matched conv1x1/final/no-horizontal-flip configuration")
    if cfg.get("physical_misalignment", {}).get("enabled", False):
        raise ValueError("Misaligned supervision cannot be combined with this law")
    if not data["strict_time_sampling"] or data["fps"] != 5 or data["dataset_fpcs"] != [64] or data["tubelet_size"] != 2:
        raise ValueError("Use strict 64f@5FPS and two-frame tubelets")
    if data.get("num_clips", 1) != 1 or data.get("dataset_type") != "CoVLAVideoDataset":
        raise ValueError("Only one CoVLA clip per sample is supported")
    if not data.get("require_encoder_train_split") or data.get("expected_encoder_train_videos") != 6305:
        raise ValueError("Require the frozen 6305-video encoder split")
    if cfg["lambda_phys"] <= 0 or not math.isfinite(float(law["alpha"])) or law["alpha"] < 0:
        raise ValueError("Keep a positive state anchor and nonnegative finite law alpha")
    if law.get("protocol") != PROTOCOL or law.get("smooth_l1_beta") != .5:
        raise ValueError("Unrecognized law protocol or beta")
    gate = json.loads(Path(law["gate_path"]).read_text())
    if gate["protocol"] != PROTOCOL or not gate["passed"] or gate["video_count"] != 6305:
        raise ValueError("A passed train-only GT consistency gate is required")
    if sha256_file(ROOT / data["encoder_train_split"]) != gate["split_sha256"]:
        raise ValueError("Law gate and encoder split differ")
    if sha256_file(cfg["stats_cache"]) != gate["state_stats_sha256"]:
        raise ValueError("State normalizer changed after the law gate")
    for name, expected in gate["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Gate implementation changed after scale freeze: {name}")
    stats = json.loads(Path(cfg["stats_cache"]).read_text())
    return gate, stats


class LawLoader:
    """Capture after worker collation, so prefetch cannot change the loss batch."""
    def __init__(self, loader, runtime):
        self.loader, self.runtime = loader, runtime

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        for sample in self.loader:
            self.runtime.batch_metadata = [entry[0][3] for entry in sample]
            self.runtime.batch_records = []
            self.runtime.loss_calls = 0
            yield sample


class LawRuntime:
    def __init__(self, args, gate, stats):
        self.args, self.gate = args, gate
        self.law = args["explicit_physics_law"]
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(stats["std"], dtype=torch.float32)
        self.rank = int(os.environ.get("RANK", "0"))
        self.folder = Path(args["folder"])
        self.folder.mkdir(parents=True, exist_ok=True)
        self.batch_metadata, self.batch_records = [], []
        self.loss_calls, self.steps, self.updates = 0, 0, 0
        self.monitored_parameters = {}
        self.last_grad = {}
        self.last_change = {}
        self.hooks = []
        self.original_state_loss = base_train.masked_smooth_l1_loss
        self.original_timer = base_train.gpu_timer
        self.original_init_opt = base_train.init_opt
        self.manifest = {
            "protocol": PROTOCOL, "config": copy.deepcopy(args), "gate": gate,
            "state_normalizer": copy.deepcopy(stats),
            "gate_sha256": sha256_file(self.law["gate_path"]),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {name: sha256_file(ROOT / name) for name in (
                "app/vjepa_2_1/train.py", "app/vjepa_2_1_law/train.py",
                "covla_law_rjob/train_from_yaml.py", "covla_law_rjob/submit.sh",
                "src/time_jepa/covla_law.py", "src/time_jepa/covla_law_dataset.py",
                "src/time_jepa/covla_time_dataset.py", "src/time_jepa/physical_aux.py")},
            "base_loss_phys_log_meaning": "state_loss + alpha * law_loss",
        }
        if self.rank == 0:
            write_json(self.folder / "law_manifest.json", self.manifest)

    def state_and_law(self, prediction, target, valid, beta):
        if self.loss_calls >= len(self.batch_metadata):
            raise RuntimeError("Physical-loss call has no matching law batch")
        metadata = self.batch_metadata[self.loss_calls]
        self.loss_calls += 1
        # Verify identity/alignment, not just matching tensor dimensions.
        expected_target = (metadata["target"][0].to(target.device) - self.mean.to(target.device)) / self.std.to(target.device)
        if not torch.equal(target, expected_target) or not torch.equal(valid, metadata["valid"][0].to(valid.device)):
            raise RuntimeError("Law metadata is not aligned to this physical prediction batch")
        state_loss = self.original_state_loss(prediction, target, valid, beta=beta)
        law_loss, record = longitudinal_law_loss(
            prediction, self.mean, self.std, metadata["law_dt"][0], metadata["law_valid"][0],
            scale=self.gate["delta_v_scale_mps"], beta=self.law["smooth_l1_beta"],
        )
        record["state_loss"] = float(state_loss.detach())
        if self.steps < int(self.law.get("health_check_steps", 10)):
            gradient = torch.autograd.grad(law_loss, prediction, retain_graph=True)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Nonfinite law gradient at state-head predictions")
            record["law_prediction_grad_norm"] = float(gradient.float().norm())
        self.batch_records.append(record)
        # alpha=0 is an exact loss ablation; keep the original tensor/value.
        return state_loss if self.law["alpha"] == 0 else state_loss + self.law["alpha"] * law_loss

    def init_opt(self, *args, **kwargs):
        result = self.original_init_opt(*args, **kwargs)
        encoder = kwargs["encoder"]
        head = kwargs["extra_modules"][0]
        self.monitored_parameters = {
            "encoder": next(p for name, p in encoder.named_parameters()
                            if "patch_embed" in name and p.requires_grad and p.ndim > 1),
            "state_head": next(p for p in head.parameters() if p.requires_grad and p.ndim > 1),
        }
        self.hooks = [result[0].register_step_pre_hook(self.before_update), result[0].register_step_post_hook(self.after_update)]
        return result

    def before_update(self, optimizer, args, kwargs):
        self.before = {}
        for name, parameter in self.monitored_parameters.items():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Missing/nonfinite {name} gradient")
            self.last_grad[name] = float(parameter.grad.float().norm())
            self.before[name] = parameter.detach().flatten()[:1024].clone()

    def after_update(self, optimizer, args, kwargs):
        self.updates += 1
        for name, parameter in self.monitored_parameters.items():
            self.last_change[name] = float((parameter.detach().flatten()[:1024] - self.before[name]).float().abs().max())

    def timed_train_step(self, closure, **kwargs):
        previous_updates = self.updates
        result, elapsed = self.original_timer(closure, **kwargs)
        self.steps += 1
        if not self.batch_records or self.loss_calls != len(self.batch_metadata):
            raise RuntimeError("Training step did not consume every law metadata batch exactly once")
        total, jepa, combined, mae, weight, lr, wd, run_step = result
        if not all(math.isfinite(x) for x in (total, jepa, combined, mae, weight, lr, wd)):
            raise FloatingPointError("Nonfinite Time-JEPA + Law training step")
        def average(key):
            return sum(r[key] for r in self.batch_records) / len(self.batch_records)
        record = {"step": self.steps, "optimizer_updates": self.updates,
                  "loss": total, "jepa_loss": jepa, "combined_physical_loss": combined,
                  "state_loss": average("state_loss"), "law_loss": average("law_loss"),
                  "law_residual_mae_mps": average("law_residual_mae_mps"),
                  "law_valid_pairs": sum(r["law_valid_pairs"] for r in self.batch_records),
                  "law_total_pairs": sum(r["law_total_pairs"] for r in self.batch_records),
                  "state_weight": weight, "law_weight": weight * self.law["alpha"],
                  "lr": lr, "gpu_time_ms": elapsed, "gradient_norms": self.last_grad.copy(),
                  "parameter_max_changes": self.last_change.copy(),
                  "updated": self.updates > previous_updates,
                  "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
        if "law_prediction_grad_norm" in self.batch_records[0]:
            record["law_prediction_grad_norm"] = average("law_prediction_grad_norm")
        if not math.isclose(combined, record["state_loss"] + self.law["alpha"] * record["law_loss"], rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError("Logged combined loss disagrees with state + alpha * law")
        if not math.isclose(total, jepa + weight * combined, rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError("Law loss did not enter the JEPA training objective as configured")
        with (self.folder / f"law_train_rank{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if self.steps <= int(self.law.get("health_check_steps", 10)) or self.steps % 10 == 0:
            logger.info("LAW_TRAIN rank=%d %s", self.rank, json.dumps(record))
        health_steps = int(self.law.get("health_check_steps", 10))
        if self.steps == health_steps:
            healthy = (self.updates == self.steps and record["law_valid_pairs"] > 0
                       and record.get("law_prediction_grad_norm", 0) > 0
                       and all(v > 0 for v in self.last_grad.values())
                       and all(v > 0 for v in self.last_change.values()))
            write_json(self.folder / f"law_health_rank{self.rank}.json", {
                "passed": healthy, "rank": self.rank, "checked_steps": self.steps,
                "training_continues": not bool(self.law.get("smoke_max_steps", 0)), "last_step": record,
            })
            if not healthy:
                raise RuntimeError("Law training health checks failed")
        if self.law.get("smoke_max_steps", 0) and self.steps >= self.law["smoke_max_steps"]:
            raise SmokeComplete()
        return result, elapsed


class SmokeComplete(Exception):
    """Optional bounded engineering run; formal configs leave this disabled."""


class TorchWithLawCheckpoints:
    """Add self-contained law provenance only to this app's checkpoints."""
    def __init__(self, runtime):
        self.runtime = runtime

    def __getattr__(self, name):
        return getattr(torch, name)

    def save(self, value, path, *args, **kwargs):
        if isinstance(value, dict) and "encoder" in value and "time_jepa" in value:
            value = dict(value, explicit_physics_law=self.runtime.manifest)
        temporary = str(path) + f".tmp.{os.getpid()}"
        torch.save(value, temporary, *args, **kwargs)
        os.replace(temporary, path)


def main(args, resume_preempt=False):
    gate, stats = validate_config(args)
    actual_world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    expected_world = int(args["explicit_physics_law"].get("expected_world_size", 8))
    if actual_world != expected_world:
        raise RuntimeError(f"Expected one {expected_world}-rank training job, got world_size={actual_world}")
    runtime = LawRuntime(args, gate, stats)
    original_init_data = base_train.init_data
    originals = {name: getattr(base_train, name) for name in (
        "init_data", "masked_smooth_l1_loss", "init_opt", "gpu_timer", "torch",
        "get_or_compute_physical_stats")}
    original_dataset = covla_time_dataset.CoVLATimeJEPADataset

    def init_data(*call_args, **kwargs):
        loader, sampler = original_init_data(*call_args, **kwargs)
        expected = set(json.loads((ROOT / args["data"]["encoder_train_split"]).read_text()))
        actual = loader.dataset.video_ids
        if len(actual) != 6305 or set(actual) != expected:
            raise ValueError("Loader must contain exactly the frozen encoder-train videos")
        if not isinstance(loader.dataset, CoVLALawDataset):
            raise TypeError("Law dataset adapter was not installed")
        return LawLoader(loader, runtime), sampler

    def frozen_stats(**kwargs):
        # The historical cache uses video_id_sha1; the newer generic stats
        # loader expects video_ids_sha1 and would silently recompute/overwrite
        # it. Preserve the actual matched baseline normalizer, verified above.
        expected = set(json.loads((ROOT / args["data"]["encoder_train_split"]).read_text()))
        actual = [row["video_id"] for row in kwargs["rows"]]
        if len(actual) != 6305 or set(actual) != expected:
            raise ValueError("Frozen normalizer requested for a different video pool")
        if normalize_target_fields(kwargs["target_fields"]) != normalize_target_fields(args["time_jepa"]["targets"]):
            raise ValueError("Frozen normalizer target order mismatch")
        device = kwargs.get("device", torch.device("cpu"))
        return runtime.mean.to(device), runtime.std.to(device)

    covla_time_dataset.CoVLATimeJEPADataset = CoVLALawDataset
    base_train.init_data = init_data
    base_train.masked_smooth_l1_loss = runtime.state_and_law
    base_train.init_opt = runtime.init_opt
    base_train.gpu_timer = runtime.timed_train_step
    base_train.torch = TorchWithLawCheckpoints(runtime)
    base_train.get_or_compute_physical_stats = frozen_stats
    try:
        base_train.main(args=copy.deepcopy(args), resume_preempt=resume_preempt)
    except SmokeComplete:
        logger.info("Bounded law smoke completed after %d optimizer steps", runtime.updates)
    finally:
        for name, value in originals.items():
            setattr(base_train, name, value)
        covla_time_dataset.CoVLATimeJEPADataset = original_dataset
        for hook in runtime.hooks:
            hook.remove()
