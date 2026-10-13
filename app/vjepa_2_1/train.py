# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    except Exception:
        pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
from app.vjepa_2_1.models.utils.masks_dist import compute_mask_distance
from app.vjepa_2_1.models.utils.modules import Lambda_LinearWarmupHold
from app.vjepa_2_1.transforms import make_transforms
from app.vjepa_2_1.utils import (
    init_opt,
    init_video_model,
    load_checkpoint,
    normalize_nested,
)
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.time_jepa.covla_time_dataset import TimeJEPAMaskCollator
from src.time_jepa.physical_aux import (
    SUPPORTED_PHYSICAL_HEAD_TYPES,
    build_physical_head,
    masked_l1_metric,
    masked_smooth_l1_loss,
)
from src.time_jepa.multidomain import (
    build_domain_epoch_schedule,
    compute_domain_step_quotas,
    domain_root_paths,
    domain_target_fields,
    get_or_compute_domain_stats,
    normalize_domain_sampling_config,
    normalize_domains_config,
    normalize_time_jepa_mode,
    resolve_domain_epoch_size,
    sample_domain_name,
)
from src.time_jepa.physical_targets import (
    get_or_compute_physical_stats,
    normalize_target_fields,
    read_covla_rows,
)
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from torch.nn.parallel import DistributedDataParallel


log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
MAX_REPEAT_COUNTS = 10

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    use_tensorboard = cfgs_meta.get("tensorboard", True)
    use_tqdm = cfgs_meta.get("tqdm", True)
    logger.info(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD')}")
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    is_causal = cfgs_model.get("is_causal", False)
    pred_is_causal = cfgs_model.get("pred_is_causal", False)
    init_type = cfgs_model.get("init_type", "default")
    img_temporal_dim_size = cfgs_model.get("img_temporal_dim_size", None)
    n_registers = cfgs_model.get("n_registers", 0)
    has_cls_first = cfgs_model.get("has_cls_first", False)
    interpolate_rope = cfgs_model.get("interpolate_rope", False)
    lambda_value_img = cfgs_model.get("lambda_value_img", 0.0)
    lambda_value_vid = cfgs_model.get("lambda_value_vid", 0.0)
    n_registers_predictor = cfgs_model.get("n_registers_predictor", 0)
    lambda_progressive = cfgs_model.get("lambda_progressive", True)
    normalize_predictor = cfgs_model.get("normalize_predictor", False)
    modality_embedding = cfgs_model.get("modality_embedding", False)
    levels_predictor = cfgs_model.get("levels_predictor", 4)
    if model_name == "vit_base":
        embed_dim_encoder = 768
    elif model_name == "vit_large":
        embed_dim_encoder = 1024
    elif model_name == "vit_giant_xformers":
        embed_dim_encoder = 1408
    elif model_name == "vit_gigantic_xformers":
        embed_dim_encoder = 1664
    else:
        print("Model name not recognized :(")

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    grid_size = crop_size // patch_size
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)

    # -- IMG DATA
    cfgs_img_data = args.get("img_data")
    img_rank_ratio = 0.25
    img_mask = None
    if cfgs_img_data is not None:
        img_dataset_type = cfgs_img_data.get("dataset_type", "imagenet")
        img_dataset_paths = cfgs_img_data.get("datasets", [])
        img_dataset_weights = cfgs_img_data.get("datasets_weights", [])
        img_dataset_fpcs = cfgs_img_data.get("dataset_fpcs")
        img_dataset_batch_size = cfgs_img_data.get("batch_size")
        img_rank_ratio = cfgs_img_data.get("rank_ratio", img_rank_ratio)
        img_num_workers = cfgs_img_data.get("num_workers", num_workers)

        img_mask = args.get("img_mask", img_mask)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    shift_by_n = cfgs_loss.get("shift_by_n")
    predict_all = cfgs_loss.get("predict_all", True)
    weight_distance_loss = cfgs_loss.get("weight_distance_loss", False)
    offset_context_loss = cfgs_loss.get("offset_context_loss", False)

    # -- TIME-JEPA AUXILIARY PHYSICAL SUPERVISION
    cfgs_time_jepa = args.get("time_jepa", {}) or {}
    time_jepa_enabled = bool(cfgs_time_jepa.get("enabled", False))
    time_jepa_mode = normalize_time_jepa_mode(cfgs_time_jepa)
    time_jepa_multidomain = time_jepa_mode == "multi_domain"
    _, time_jepa_domains = normalize_domains_config(cfgs_time_jepa)
    domain_sampling_cfg = normalize_domain_sampling_config(
        cfgs_time_jepa,
        default_seed=seed,
    )
    time_jepa_targets = normalize_target_fields(
        cfgs_time_jepa.get("targets", ["vEgo", "aEgo", "yaw_rate"])
    )
    lambda_phys = float(cfgs_time_jepa.get("lambda_phys", 0.0))
    lambda_phys_warmup_steps = int(
        cfgs_time_jepa.get(
            "lambda_phys_warmup_steps",
            cfgs_time_jepa.get("lambda_phys_warmup", 0),
        )
    )
    phys_loss_beta = float(cfgs_time_jepa.get("smooth_l1_beta", 1.0))
    phys_stats_cache = cfgs_time_jepa.get("stats_cache", None)
    phys_stats_max_videos = int(cfgs_time_jepa.get("stats_max_videos", 2000))
    phys_stats_seed = int(cfgs_time_jepa.get("stats_seed", seed))
    disable_flip_default = False
    if time_jepa_multidomain and any(cfg["domain"] == "agibot" for cfg in time_jepa_domains.values()):
        disable_flip_default = True
    disable_flip_for_phys = bool(cfgs_time_jepa.get("disable_horizontal_flip", disable_flip_default))
    physical_head_type = str(cfgs_time_jepa.get("head_type", "linear")).strip().lower()
    physical_head_hidden_dim = int(cfgs_time_jepa.get("head_hidden_dim", 256))
    physical_head_dilations_cfg = cfgs_time_jepa.get("head_temporal_dilations", [1, 2, 4])
    if physical_head_dilations_cfg is None:
        physical_head_dilations = (1, 2, 4)
    elif isinstance(physical_head_dilations_cfg, str):
        physical_head_dilations = tuple(
            int(value.strip())
            for value in physical_head_dilations_cfg.split(",")
            if value.strip()
        )
    elif isinstance(physical_head_dilations_cfg, (int, float)):
        physical_head_dilations = (int(physical_head_dilations_cfg),)
    else:
        physical_head_dilations = tuple(int(value) for value in physical_head_dilations_cfg)
    physical_feature_mode = str(cfgs_time_jepa.get("feature_mode", "final")).strip().lower()
    phys_misalignment_cfg = cfgs_time_jepa.get("physical_misalignment", {}) or {}
    if isinstance(phys_misalignment_cfg, bool):
        phys_misalignment_enabled = phys_misalignment_cfg
        phys_misalignment_cfg = {}
    else:
        phys_misalignment_enabled = bool(phys_misalignment_cfg.get("enabled", False))
    phys_misalignment_mode = str(
        phys_misalignment_cfg.get(
            "mode",
            "derangement" if phys_misalignment_enabled else "none",
        )
    )
    if not phys_misalignment_enabled:
        phys_misalignment_mode = "none"
    phys_misalignment_seed = int(phys_misalignment_cfg.get("seed", seed))
    phys_misalignment_shift = int(phys_misalignment_cfg.get("shift", 1))
    if time_jepa_enabled and (not time_jepa_multidomain) and lambda_phys <= 0:
        raise ValueError("time_jepa.enabled requires time_jepa.lambda_phys > 0")
    if time_jepa_enabled and physical_head_type not in SUPPORTED_PHYSICAL_HEAD_TYPES:
        supported_head_types = ", ".join(SUPPORTED_PHYSICAL_HEAD_TYPES)
        raise ValueError(f"time_jepa.head_type must be one of: {supported_head_types}")
    if time_jepa_enabled and physical_feature_mode not in ("final", "hierarchical"):
        raise ValueError("time_jepa.feature_mode must be one of: final, hierarchical")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    resume_anneal = cfgs_opt.get("resume_anneal", False) or (
        is_anneal and resume_preempt
    )
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    use_radamw = cfgs_opt.get("use_radamw", False)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    loss_reg_std_mult = cfgs_opt.get("loss_reg_std_mult", None)
    loss_reg_num_tracking_steps = cfgs_opt.get("loss_reg_num_tracking_steps", 300)
    loss_reg_min_epoch = cfgs_opt.get("loss_reg_min_epoch", 50)
    if loss_reg_std_mult is not None:
        logger.info("Loss regulation activated")
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    data_world_size, data_rank = world_size, rank
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    img_world_size = 0

    # make adjustments to batch size for image data
    model_fpcs = dataset_fpcs
    model_cfgs_mask = cfgs_mask
    model_tubelet_size = tubelet_size
    if cfgs_img_data is not None:
        img_world_size = int(world_size * img_rank_ratio)
        num_video_ranks = world_size - img_world_size
        img_total_batch_size = img_dataset_batch_size * world_size
        video_total_batch_size = batch_size * world_size

        if img_total_batch_size % img_world_size != 0:
            raise ValueError(
                f"img_total_batch_size ({img_total_batch_size}) must be divisible by num_img_ranks ({img_world_size})"
            )
        if video_total_batch_size % num_video_ranks != 0:
            raise ValueError(
                f"video_total_batch_size ({video_total_batch_size}) must be divisible by num_video_ranks ({num_video_ranks})"
            )

        # img_dataset_batch_size = img_total_batch_size // img_world_size
        batch_size = video_total_batch_size // num_video_ranks

        if rank < int(world_size * img_rank_ratio):
            crop_size = cfgs_img_data.get("crop_size", 512)
            grid_size = crop_size // patch_size

        if rank < int(world_size * img_rank_ratio):
            logger.info(
                f"On rank {rank}, updating dataset with dataset type {img_dataset_type}"
            )
            if img_temporal_dim_size is not None:
                if img_dataset_fpcs[0] != 1:
                    raise NotImplementedError(
                        "Image loader only supports 1 frame per clip with img_temporal_dim_size=1"
                    )
                tubelet_size = 1
            else:
                tubelet_size = tubelet_size

            dataset_type = img_dataset_type
            dataset_paths = img_dataset_paths
            datasets_weights = img_dataset_weights
            dataset_fpcs = img_dataset_fpcs
            batch_size = img_dataset_batch_size
            num_workers = img_num_workers
            if img_mask is not None:
                logger.info("Using image mask")
                cfgs_mask = img_mask

            data_rank = rank
            data_world_size = img_world_size
            lambda_value = lambda_value_img  # We select a different lambda value depending on video vs. image
        else:
            data_rank = rank - img_world_size
            data_world_size = world_size - img_world_size
            lambda_value = lambda_value_vid  # We select a different lambda value depending on video vs. image

        logger.info(
            f"For rank {rank} with world size {world_size}, "
            f"we have total image batch size {img_total_batch_size}, total video batch size {video_total_batch_size}, "
            f"image ranks: {img_world_size}, video ranks: {num_video_ranks}, "
            f"using the following params: "
            f"dataset_type: {dataset_type}, "
            f"dataset_paths: {dataset_paths}, "
            f"datasets_weights: {datasets_weights}, "
            f"dataset_fpcs: {dataset_fpcs}, "
            f"batch_size: {batch_size}, "
            f"num_workers: {num_workers}, "
            f"data_rank: {data_rank}, "
            f"data_world_size: {data_world_size}"
            f"lambda_value for the context loss: {lambda_value}"
        )
    else:
        lambda_value = lambda_value_vid

    if time_jepa_multidomain:
        dataset_type = "multi_domain_time_jepa"
        for domain_name, domain_cfg in time_jepa_domains.items():
            domain_fpcs = domain_cfg.get("dataset_fpcs", dataset_fpcs)
            domain_fps = domain_cfg.get("fps", fps)
            if list(domain_fpcs) != list(dataset_fpcs):
                raise ValueError(
                    f"time_jepa.domains.{domain_name}.dataset_fpcs must match top-level data.dataset_fpcs "
                    f"for first multi-domain version; got {domain_fpcs} vs {dataset_fpcs}"
                )
            if domain_fps != fps:
                raise ValueError(
                    f"time_jepa.domains.{domain_name}.fps must match top-level data.fps "
                    f"for strict shared time windows; got {domain_fps} vs {fps}"
                )
    elif time_jepa_enabled:
        if dataset_type.lower() not in ("covlatimejepadataset", "covla_time_jepa", "covla_time"):
            if dataset_type.lower() in ("covlavideodataset", "covla_video", "covla"):
                logger.info("time_jepa.enabled: switching CoVLA data loader to covla_time_jepa")
                dataset_type = "covla_time_jepa"
            else:
                raise ValueError("time_jepa.enabled currently requires a CoVLA video dataset")
    if (time_jepa_enabled or time_jepa_multidomain) and cfgs_img_data is not None:
        raise NotImplementedError("time_jepa.enabled/time_jepa.mode=multi_domain is currently implemented for video-only ranks")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        local_rank = local_rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pth.tar"
    latest_path = os.path.join(folder, latest_file)

    load_path = None
    if load_model:
        if is_anneal:
            if os.path.exists(latest_path) and resume_anneal:
                load_path = latest_path
            else:
                load_path = anneal_ckpt
                resume_anneal = False
        else:
            load_path = r_file if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False
    elif os.path.exists(latest_path):
        load_path = latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )
    tb_writer = None
    if rank == 0 and use_tensorboard:
        if SummaryWriter is None:
            logger.warning("TensorBoard requested but torch.utils.tensorboard is unavailable.")
        else:
            tb_dir = os.path.join(folder, "tensorboard")
            tb_writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"Writing TensorBoard events to {tb_dir}")

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(model_cfgs_mask) * len(model_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=model_tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=predict_all,
        chop_last_n_tokens=shift_by_n,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        n_registers_predictor=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
    )
    target_encoder = copy.deepcopy(encoder)
    physical_head = None
    time_heads = None
    physical_target_mean = None
    physical_target_std = None
    domain_target_mean = {}
    domain_target_std = {}
    domain_target_names = {}
    domain_lambdas = {}
    domain_betas = {}
    if time_jepa_enabled:
        if len(dataset_fpcs) != 1:
            raise NotImplementedError("time_jepa.enabled currently expects one video frames-per-clip value")
        physical_feature_levels = 1
        if physical_feature_mode == "hierarchical":
            physical_feature_levels = len(encoder.backbone.out_layers_distillation)
        physical_head_embed_dim = embed_dim_encoder * physical_feature_levels
        if time_jepa_multidomain:
            time_heads = torch.nn.ModuleDict()
            for domain_name, domain_cfg in time_jepa_domains.items():
                targets = domain_target_fields(domain_cfg)
                domain_target_names[domain_name] = targets
                domain_lambdas[domain_name] = float(domain_cfg.get("lambda_time", 0.0))
                domain_betas[domain_name] = float(domain_cfg.get("smooth_l1_beta", phys_loss_beta))
                if domain_lambdas[domain_name] <= 0:
                    raise ValueError(f"time_jepa.domains.{domain_name}.lambda_time must be > 0")
                time_heads[domain_name] = build_physical_head(
                    head_type=physical_head_type,
                    embed_dim=physical_head_embed_dim,
                    num_outputs=len(targets),
                    num_frames=dataset_fpcs[0],
                    tubelet_size=tubelet_size,
                    hidden_dim=physical_head_hidden_dim,
                    temporal_dilations=physical_head_dilations,
                )
            time_heads = time_heads.to(device)
            logger.info("Multi-domain Time-JEPA domains: %s", sorted(time_jepa_domains))
            for domain_name in sorted(time_jepa_domains):
                logger.info(
                    "Time-JEPA domain=%s targets=%s lambda=%s beta=%s",
                    domain_name,
                    domain_target_names[domain_name],
                    domain_lambdas[domain_name],
                    domain_betas[domain_name],
                )
        else:
            physical_head = build_physical_head(
                head_type=physical_head_type,
                embed_dim=physical_head_embed_dim,
                num_outputs=len(time_jepa_targets),
                num_frames=dataset_fpcs[0],
                tubelet_size=tubelet_size,
                hidden_dim=physical_head_hidden_dim,
                temporal_dilations=physical_head_dilations,
            ).to(device)
            logger.info("Time-JEPA physical targets: %s", time_jepa_targets)
        logger.info(
            "Time-JEPA physical feature mode: %s embed_dim=%s levels=%s",
            physical_feature_mode,
            physical_head_embed_dim,
            physical_feature_levels,
        )
        logger.info(
            "Time-JEPA physical head: %s hidden_dim=%s temporal_dilations=%s",
            time_heads if time_heads is not None else physical_head,
            physical_head_hidden_dim,
            physical_head_dilations,
        )
        logger.info(
            "Time-JEPA physical misalignment: enabled=%s mode=%s seed=%s shift=%s",
            phys_misalignment_enabled,
            phys_misalignment_mode,
            phys_misalignment_seed,
            phys_misalignment_shift,
        )

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()
        if physical_head is not None:
            physical_head.compile()
        if time_heads is not None:
            for domain_name in time_heads:
                time_heads[domain_name].compile()

    collator_cls = TimeJEPAMaskCollator if time_jepa_enabled else MaskCollator
    mask_collator = collator_cls(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    random_horizontal_flip = not disable_flip_for_phys
    transform = make_transforms(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
        return_metadata=time_jepa_enabled and random_horizontal_flip,
    )
    logger.info(
        "Training horizontal flip: enabled=%s (time_jepa.disable_horizontal_flip=%s)",
        random_horizontal_flip,
        disable_flip_for_phys,
    )

    if time_jepa_enabled and time_jepa_multidomain:
        for domain_name, domain_cfg in time_jepa_domains.items():
            stats_cache = domain_cfg.get("stats_cache")
            if stats_cache is None:
                stats_cache = os.path.join(folder, f"time_jepa_{domain_name}_stats.json")
                domain_cfg["stats_cache"] = stats_cache
            mean, std = get_or_compute_domain_stats(
                domain_cfg=domain_cfg,
                frames_per_clip=dataset_fpcs[0],
                tubelet_size=tubelet_size,
                fps=fps,
                frame_step=None,
                duration=None,
                num_clips=1,
                random_clip_sampling=True,
                allow_clip_overlap=False,
                strict_time_sampling=bool(domain_cfg.get("strict_time_sampling", True)),
                cache_path=stats_cache,
                distributed=world_size > 1,
                rank=rank,
                device=device,
            )
            domain_target_mean[domain_name] = mean.to(device)
            domain_target_std[domain_name] = std.to(device)
            logger.info(
                "Time-JEPA domain=%s stats cache=%s mean=%s std=%s",
                domain_name,
                stats_cache,
                domain_target_mean[domain_name].detach().cpu().tolist(),
                domain_target_std[domain_name].detach().cpu().tolist(),
            )
    elif time_jepa_enabled:
        if phys_stats_cache is None:
            phys_stats_cache = os.path.join(folder, "time_jepa_physical_stats.json")
        stats_rows = read_covla_rows(dataset_paths)
        physical_target_mean, physical_target_std = get_or_compute_physical_stats(
            cache_path=phys_stats_cache,
            rows=stats_rows,
            target_fields=time_jepa_targets,
            frames_per_clip=dataset_fpcs[0],
            tubelet_size=tubelet_size,
            frame_step=None,
            fps=fps,
            duration=None,
            num_clips=1,
            random_clip_sampling=True,
            allow_clip_overlap=False,
            seed=phys_stats_seed,
            max_videos=phys_stats_max_videos,
            strict_time_sampling=bool(cfgs_data.get("strict_time_sampling", False)),
            distributed=world_size > 1,
            rank=rank,
            device=device,
        )
        physical_target_mean = physical_target_mean.to(device)
        physical_target_std = physical_target_std.to(device)
        logger.info(
            "Time-JEPA physical stats cache=%s mean=%s std=%s",
            phys_stats_cache,
            physical_target_mean.detach().cpu().tolist(),
            physical_target_std.detach().cpu().tolist(),
        )

    # -- init data-loaders/samplers
    domain_loaders = {}
    domain_samplers = {}
    domain_loader_lengths = {}
    if time_jepa_multidomain:
        for domain_name, domain_cfg in time_jepa_domains.items():
            domain_loader, domain_sampler = init_data(
                data=domain_cfg.get("dataset_type"),
                root_path=domain_root_paths(domain_cfg),
                batch_size=int(domain_cfg.get("batch_size", batch_size)),
                training=True,
                dataset_fpcs=domain_cfg.get("dataset_fpcs", dataset_fpcs),
                fps=domain_cfg.get("fps", fps),
                transform=transform,
                rank=data_rank,
                world_size=data_world_size,
                datasets_weights=domain_cfg.get("datasets_weights"),
                collator=mask_collator,
                num_workers=int(domain_cfg.get("num_workers", num_workers)),
                pin_mem=bool(domain_cfg.get("pin_mem", pin_mem)),
                persistent_workers=bool(domain_cfg.get("persistent_workers", cfgs_data.get("persistent_workers", False))),
                log_dir=None,
                physical_target_fields=domain_target_names.get(domain_name) if time_jepa_enabled else None,
                physical_tubelet_size=tubelet_size,
                physical_misalignment_enabled=(
                    phys_misalignment_enabled if time_jepa_enabled and domain_cfg["domain"] == "covla" else False
                ),
                physical_misalignment_mode=phys_misalignment_mode,
                physical_misalignment_seed=phys_misalignment_seed,
                physical_misalignment_shift=phys_misalignment_shift,
                strict_time_sampling=bool(domain_cfg.get("strict_time_sampling", True)),
                agibot_tasks=domain_cfg.get("tasks"),
                agibot_video_key=domain_cfg.get("video_key", "observation.images.top_head"),
                agibot_target_fields=domain_cfg.get("targets") if time_jepa_enabled else None,
                agibot_slice_mode=domain_cfg.get("slice_mode", "random"),
                agibot_slice_stride_seconds=domain_cfg.get("slice_stride_seconds"),
                agibot_max_windows_per_episode=domain_cfg.get("max_windows_per_episode"),
                agibot_use_meta_episodes_only=bool(domain_cfg.get("use_meta_episodes_only", True)),
                agibot_max_time_delta=domain_cfg.get("max_time_delta"),
                agibot_video_backend=domain_cfg.get("video_backend", "auto"),
                agibot_ffmpeg_timeout=domain_cfg.get("ffmpeg_timeout", 120),
                return_physical_targets=time_jepa_enabled,
                agibot_episode_allowlist=domain_cfg.get("episode_allowlist"),
                agibot_view_episode_allowlists=domain_cfg.get(
                    "view_episode_allowlists"
                ),
                egoexo_video_view=domain_cfg.get("video_views", domain_cfg.get("video_view", "best_exo")),
                egoexo_target_fields=domain_cfg.get("targets") if time_jepa_enabled else None,
                egoexo_slice_stride_seconds=domain_cfg.get("slice_stride_seconds", 12.8),
                egoexo_max_time_delta=domain_cfg.get("max_time_delta", 0.05),
                egoexo_pose_fps=domain_cfg.get("pose_fps", 30.0),
                egoexo_min_views=domain_cfg.get("min_views", 2),
                egoexo_smooth_window=domain_cfg.get("smooth_window", 5),
                egoexo_video_backend=domain_cfg.get("video_backend", "auto"),
                egoexo_ffmpeg_timeout=domain_cfg.get("ffmpeg_timeout", 120),
                egoexo_episode_allowlist=domain_cfg.get("episode_allowlist"),
                egoexo_manifest_path=domain_cfg.get("manifest_path"),
                egoexo_split_prefix=domain_cfg.get("split_prefix", "egoexo_bestexo"),
                egoexo_target_temporal_mode=domain_cfg.get(
                    "target_temporal_mode", "tubelet_aligned"
                ),
            )
            domain_loaders[domain_name] = domain_loader
            domain_samplers[domain_name] = domain_sampler
            try:
                domain_loader_lengths[domain_name] = len(domain_loader)
            except Exception:
                domain_loader_lengths[domain_name] = getattr(domain_loader, "num_batches", -1)
        unsupervised_loader = None
        unsupervised_sampler = None
        positive_lengths = [v for v in domain_loader_lengths.values() if v and v > 0]
        _dlen = min(positive_lengths) if positive_lengths else -1
    else:
        (unsupervised_loader, unsupervised_sampler) = init_data(
            data=dataset_type,
            root_path=dataset_paths,
            batch_size=batch_size,
            training=True,
            # clip_len=clip_len,
            dataset_fpcs=dataset_fpcs,
            fps=fps,
            transform=transform,
            rank=data_rank,
            world_size=data_world_size,
            datasets_weights=datasets_weights,
            collator=mask_collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            log_dir=None,
            physical_target_fields=time_jepa_targets if time_jepa_enabled else None,
            physical_tubelet_size=tubelet_size,
            physical_misalignment_enabled=phys_misalignment_enabled if time_jepa_enabled else False,
            physical_misalignment_mode=phys_misalignment_mode,
            physical_misalignment_seed=phys_misalignment_seed,
            physical_misalignment_shift=phys_misalignment_shift,
            strict_time_sampling=bool(cfgs_data.get("strict_time_sampling", False)),
        )
        try:
            _dlen = len(unsupervised_loader)
        except Exception:
            try:
                _dlen = unsupervised_loader.num_batches
            except Exception:
                _dlen = -1
    if ipe is None:
        if time_jepa_multidomain:
            ipe = resolve_domain_epoch_size(
                domain_loader_lengths,
                domain_sampling_cfg["epoch_size"],
            )
        else:
            ipe = _dlen
    ipe = int(ipe)
    if ipe <= 0:
        raise ValueError(f"iterations per epoch must be positive, got {ipe}")
    logger.info(f"Using batch size of {batch_size}, fpcs of {dataset_fpcs}")
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")
    if time_jepa_multidomain:
        logger.info("Multi-domain loader lengths: %s", domain_loader_lengths)
        logger.info(
            "Multi-domain scheduling: mode=%s epoch_size=%s resolved_ipe=%s "
            "shuffle=%s seed=%s",
            domain_sampling_cfg["mode"],
            domain_sampling_cfg["epoch_size"],
            ipe,
            domain_sampling_cfg["shuffle"],
            domain_sampling_cfg["seed"],
        )

    # zizi

    # -- init optimizer and scheduler
    extra_opt_modules = []
    if physical_head is not None:
        extra_opt_modules.append(physical_head)
    if time_heads is not None:
        extra_opt_modules.extend([time_heads[name] for name in time_heads])
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=predictor,
        extra_modules=extra_opt_modules if extra_opt_modules else None,
        use_radamw=use_radamw,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    ddp_device_ids = [device.index] if device.type == "cuda" else None
    encoder = DistributedDataParallel(
        encoder,
        static_graph=True,
        device_ids=ddp_device_ids,
        output_device=device.index if device.type == "cuda" else None,
    )
    predictor = DistributedDataParallel(
        predictor,
        static_graph=False,
        find_unused_parameters=True,
        device_ids=ddp_device_ids,
        output_device=device.index if device.type == "cuda" else None,
    )
    if physical_head is not None:
        physical_head = DistributedDataParallel(
            physical_head,
            static_graph=True,
            device_ids=ddp_device_ids,
            output_device=device.index if device.type == "cuda" else None,
        )
    if time_heads is not None:
        for domain_name in list(time_heads.keys()):
            time_heads[domain_name] = DistributedDataParallel(
                time_heads[domain_name],
                static_graph=False,
                device_ids=ddp_device_ids,
                output_device=device.index if device.type == "cuda" else None,
            )
    target_encoder = DistributedDataParallel(
        target_encoder,
        device_ids=ddp_device_ids,
        output_device=device.index if device.type == "cuda" else None,
    )
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs) + 1)
    )
    lambda_sched = Lambda_LinearWarmupHold(lambda_value=lambda_value)

    start_epoch = 0
    # -- load training checkpoint
    print("Loadind checkpoint from: ", load_path)
    if load_path is not None:
        extra_load_modules = {}
        if physical_head is not None:
            extra_load_modules["physical_head"] = physical_head
        if time_heads is not None:
            for domain_name in time_heads:
                extra_load_modules[f"time_heads.{domain_name}"] = time_heads[domain_name]
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal and not resume_anneal,
            extra_modules=extra_load_modules if extra_load_modules else None,
        )
        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        if physical_head is not None:
            save_dict["physical_head"] = physical_head.state_dict()
            save_dict["time_jepa"] = {
                "enabled": time_jepa_enabled,
                "targets": time_jepa_targets,
                "lambda_phys": lambda_phys,
                "head_type": physical_head_type,
                "head_hidden_dim": physical_head_hidden_dim,
                "head_temporal_dilations": physical_head_dilations,
                "feature_mode": physical_feature_mode,
                "stats_cache": phys_stats_cache,
                "physical_misalignment": {
                    "enabled": phys_misalignment_enabled,
                    "mode": phys_misalignment_mode,
                    "seed": phys_misalignment_seed,
                    "shift": phys_misalignment_shift,
                },
            }
        if time_heads is not None:
            save_dict["time_heads"] = {
                domain_name: time_heads[domain_name].state_dict()
                for domain_name in time_heads
            }
            save_dict["extra_modules"] = {
                f"time_heads.{domain_name}": time_heads[domain_name].state_dict()
                for domain_name in time_heads
            }
            save_dict["time_jepa"] = {
                "enabled": time_jepa_enabled,
                "mode": time_jepa_mode,
                "domains": time_jepa_domains,
                "head_type": physical_head_type,
                "head_hidden_dim": physical_head_hidden_dim,
                "head_temporal_dilations": physical_head_dilations,
                "feature_mode": physical_feature_mode,
                "lambda_warmup_steps": lambda_phys_warmup_steps,
                "domain_sampling": domain_sampling_cfg,
            }
        elif time_jepa_multidomain:
            save_dict["time_jepa"] = {
                "enabled": time_jepa_enabled,
                "mode": time_jepa_mode,
                "domains": time_jepa_domains,
                "domain_sampling": domain_sampling_cfg,
            }
        if time_jepa_multidomain:
            save_dict["domain_total_step_counts"] = dict(
                domain_total_step_counts
            )
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    if time_jepa_multidomain:
        for domain_name, sampler in domain_samplers.items():
            sampler.set_epoch(start_epoch)
        domain_iters = {domain_name: iter(loader) for domain_name, loader in domain_loaders.items()}
        domain_names = list(time_jepa_domains.keys())
        domain_weights = [float(time_jepa_domains[name].get("sample_weight", 1.0)) for name in domain_names]
        logger.info("Multi-domain sampling weights: %s", dict(zip(domain_names, domain_weights)))
        if domain_sampling_cfg["quota_basis"] == "loader_lengths":
            domain_quota_weights = [
                float(domain_loader_lengths[name]) for name in domain_names
            ]
        else:
            domain_quota_weights = domain_weights
        logger.info(
            "Multi-domain quota basis=%s effective quota weights=%s",
            domain_sampling_cfg["quota_basis"],
            dict(zip(domain_names, domain_quota_weights)),
        )
        domain_total_step_counts = {name: 0 for name in domain_names}
        if domain_sampling_cfg["mode"] == "quota":
            for completed_epoch in range(start_epoch):
                completed_quotas = compute_domain_step_quotas(
                    domain_names=domain_names,
                    weights=domain_quota_weights,
                    total_steps=ipe,
                    epoch=completed_epoch,
                )
                for domain_name in domain_names:
                    domain_total_step_counts[domain_name] += completed_quotas[
                        domain_name
                    ]

        def make_domain_schedule(epoch):
            if domain_sampling_cfg["mode"] != "quota":
                return None, None
            return build_domain_epoch_schedule(
                domain_names=domain_names,
                weights=domain_quota_weights,
                total_steps=ipe,
                epoch=epoch,
                seed=domain_sampling_cfg["seed"],
                shuffle=domain_sampling_cfg["shuffle"],
                device=device,
                distributed=world_size > 1,
                rank=rank,
            )

        domain_schedule, planned_domain_quotas = make_domain_schedule(start_epoch)
        domain_exposure_log = os.path.join(folder, "domain_exposure.jsonl")
    else:
        unsupervised_sampler.set_epoch(start_epoch)
        loader = iter(unsupervised_loader)

    if skip_batches > 0:
        if skip_batches > ipe:
            raise ValueError(
                f"meta.skip_batches={skip_batches} exceeds iterations per epoch={ipe}"
            )
        logger.info(f"Skip {skip_batches} batches")

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                if time_jepa_multidomain:
                    if domain_schedule is not None:
                        skip_domain = domain_schedule[itr]
                    else:
                        skip_domain = sample_domain_name(
                            domain_names,
                            domain_weights,
                            device=device,
                            distributed=world_size > 1,
                            rank=rank,
                        )
                    _ = next(domain_iters[skip_domain])
                else:
                    _ = next(loader)
            except Exception:
                if time_jepa_multidomain:
                    domain_iters[skip_domain] = iter(domain_loaders[skip_domain])
                    _ = next(domain_iters[skip_domain])
                else:
                    loader = iter(unsupervised_loader)
                    _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    trailing_losses = []
    step_count = 0

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        if time_jepa_multidomain:
            for domain_name, sampler in domain_samplers.items():
                sampler.set_epoch(epoch)
            if (
                domain_sampling_cfg["mode"] == "quota"
                and not (epoch == start_epoch and skip_batches > 0)
            ):
                domain_iters = {
                    domain_name: iter(loader)
                    for domain_name, loader in domain_loaders.items()
                }
            domain_schedule, planned_domain_quotas = make_domain_schedule(epoch)
            if planned_domain_quotas is not None:
                logger.info(
                    "Epoch %d planned domain step quotas: %s",
                    epoch + 1,
                    planned_domain_quotas,
                )

        loss_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()
        if time_jepa_multidomain:
            domain_step_counts = {name: 0 for name in domain_names}
            domain_sample_counts = {name: 0 for name in domain_names}
            domain_loss_meters = {name: AverageMeter() for name in domain_names}
            domain_jepa_loss_meters = {name: AverageMeter() for name in domain_names}
            domain_time_loss_meters = {name: AverageMeter() for name in domain_names}
            domain_time_mae_meters = {name: AverageMeter() for name in domain_names}

        progress = range(ipe)
        pbar = None
        if rank == 0 and use_tqdm and tqdm is not None:
            pbar = tqdm(
                progress,
                desc=f"epoch {epoch + 1}/{num_epochs}",
                dynamic_ncols=True,
                mininterval=10.0,
                leave=True,
            )
            progress = pbar

        for itr in progress:
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            active_domain = None
            if time_jepa_multidomain:
                if domain_schedule is not None:
                    active_domain = domain_schedule[itr]
                else:
                    active_domain = sample_domain_name(
                        domain_names,
                        domain_weights,
                        device=device,
                        distributed=world_size > 1,
                        rank=rank,
                    )
            while not iter_successful:
                try:
                    if time_jepa_multidomain:
                        sample = next(domain_iters[active_domain])
                    else:
                        sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    if time_jepa_multidomain:
                        domain_samplers[active_domain].set_epoch(epoch)
                        domain_iters[active_domain] = iter(domain_loaders[active_domain])
                    elif "airstore" in dataset_type.lower():
                        unsupervised_sampler.increase_epoch()
                    else:
                        unsupervised_sampler.set_epoch(epoch)
                        loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(
                            f"Encountered exception when loading data (num retries {iter_retries}):\n{e}"
                        )
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        raise RuntimeError(
                            f"Exceeded max retries ({NUM_RETRIES}) when loading data."
                        ) from e

            step_local_samples = 0
            for _fpc_sample in sample:
                bs, fpc = _fpc_sample[0][2][0].size()
                step_local_samples += int(bs)
                mask_meters[fpc].update(bs / batch_size)
            if time_jepa_multidomain:
                domain_step_counts[active_domain] += 1
                domain_total_step_counts[active_domain] += 1
                domain_sample_counts[active_domain] += (
                    step_local_samples * int(data_world_size)
                )

            def load_clips():
                all_clips, all_masks_enc, all_masks_pred = [], [], []
                physical_batches = []
                for fpc_sample in sample:
                    udata, masks_enc, masks_pred = fpc_sample
                    all_clips += [udata[0][0].to(device, non_blocking=True)]
                    all_masks_enc += [
                        [m.to(device, non_blocking=True) for m in masks_enc]
                    ]
                    all_masks_pred += [
                        [m.to(device, non_blocking=True) for m in masks_pred]
                    ]
                    if time_jepa_enabled:
                        phys = udata[3]
                        physical_batches.append(
                            {
                                "target": phys["target"][0].to(device, non_blocking=True),
                                "valid": phys["valid"][0].to(device, non_blocking=True),
                                "domain": active_domain if active_domain is not None else "covla",
                            }
                        )
                return all_clips, all_masks_enc, all_masks_pred, physical_batches

            clips, masks_enc, masks_pred, physical_batches = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(c, embed_dim=embed_dim_encoder):
                    with torch.no_grad():
                        h = target_encoder(c, gram_mode=False, training_mode=True)
                        new_h = []
                        for hi in h:
                            if levels_predictor > 1:
                                hi_0 = F.layer_norm(hi[:, :, :embed_dim], (embed_dim,))
                                hi_1 = F.layer_norm(
                                    hi[:, :, embed_dim : embed_dim * 2],
                                    (embed_dim,),
                                )
                                hi_2 = F.layer_norm(
                                    hi[:, :, embed_dim * 2 : embed_dim * 3],
                                    (embed_dim,),
                                )
                                hi_3 = F.layer_norm(hi[:, :, -embed_dim:], (embed_dim,))
                                hi_norm = torch.cat([hi_0, hi_1, hi_2, hi_3], dim=2)
                                new_h.append(hi_norm)
                            else:
                                new_h.append(F.layer_norm(hi, (hi.size(-1),)))
                        return new_h

                def forward_context(clips, embed_dim=embed_dim_encoder):
                    modality = "video"
                    if img_temporal_dim_size is not None:
                        if clips[0].shape[2] == img_temporal_dim_size:
                            modality = "image"
                    z = encoder(clips, masks_enc, gram_mode=False, training_mode=True)
                    z_pred, z_context = predictor(
                        z, masks_enc, masks_pred, mod=modality
                    )
                    if normalize_predictor:
                        z_pred = normalize_nested(z_pred, embed_dim)

                        if predict_all:
                            z_context = normalize_nested(z_context, embed_dim)
                    return z_pred, z_context

                def forward_physical(clips, domain_name=None):
                    if not time_jepa_enabled:
                        return None, None
                    z_full = encoder(
                        clips,
                        None,
                        gram_mode=False,
                        training_mode=physical_feature_mode == "hierarchical",
                    )
                    preds = []
                    head = physical_head
                    if time_heads is not None:
                        head = time_heads[domain_name]
                    for zi in z_full:
                        preds.append(head(zi))
                    return preds, z_full

                def loss_fn(z, h, masks_to_apply, cls_loss, d_weights):
                    if cls_loss:
                        h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
                        h = [
                            apply_masks(hi[:, 1:], mi, concat=False)
                            for hi, mi in zip(h, masks_to_apply)
                        ]
                        loss, n = 0, 0
                        for zi, hi, hi_cls in zip(z, h, h_cls):
                            for zij, hij in zip(zi, hi):
                                h_term = torch.cat([hi_cls, hij], dim=1)
                                loss += (
                                    torch.mean(torch.abs(zij - h_term) ** loss_exp)
                                    / loss_exp
                                )
                                n += 1

                        loss /= n
                        return loss
                    else:
                        h = [
                            apply_masks(hi, mi, concat=False)
                            for hi, mi in zip(h, masks_to_apply)
                        ]

                        if d_weights is not None:
                            loss, n = 0, 0
                            for zi, hi, d_i in zip(z, h, d_weights):
                                for zij, hij, d_ij in zip(zi, hi, d_i):
                                    loss_n = torch.abs(zij - hij) ** loss_exp * (
                                        1 / d_ij.unsqueeze(2)
                                    )
                                    loss += torch.mean(loss_n) / loss_exp
                                    n += 1
                            loss /= n
                            return loss
                        else:
                            loss, n = 0, 0
                            for zi, hi in zip(z, h):
                                for zij, hij in zip(zi, hi):
                                    loss += (
                                        torch.mean(torch.abs(zij - hij) ** loss_exp)
                                        / loss_exp
                                    )
                                    n += 1
                            loss /= n
                            return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z_pred, z_context = forward_context(clips)
                    loss = 0
                    loss_pred = loss_fn(
                        z_pred, h, masks_pred, cls_loss=has_cls_first, d_weights=None
                    )
                    loss += loss_pred

                    # Context loss
                    if predict_all:
                        distance_weights = compute_mask_distance(
                            masks_pred, masks_enc, grid_size, offset_context_loss
                        )
                        if weight_distance_loss:
                            d_weights = distance_weights
                        else:
                            d_weights = None
                        loss_context = loss_fn(
                            z_context, h, masks_enc, cls_loss=False, d_weights=d_weights
                        )
                        if lambda_progressive:
                            lambda_value_step = lambda_sched.value(epoch * ipe + itr)
                        else:
                            lambda_value_step = lambda_value
                        loss += loss_context * lambda_value_step

                    loss_jepa = loss
                    loss_phys = torch.zeros((), device=device, dtype=loss.dtype)
                    phys_mae_norm = torch.zeros((), device=device, dtype=loss.dtype)
                    lambda_phys_step = 0.0
                    if time_jepa_enabled:
                        if len(physical_batches) != len(clips):
                            raise RuntimeError(
                                f"Expected {len(clips)} physical target batches, got {len(physical_batches)}"
                            )
                        loss_domain = active_domain if time_jepa_multidomain else None
                        phys_preds, _ = forward_physical(clips, domain_name=loss_domain)
                        phys_losses, phys_maes = [], []
                        if time_jepa_multidomain:
                            mean = domain_target_mean[loss_domain].view(1, 1, -1)
                            std = domain_target_std[loss_domain].view(1, 1, -1)
                            beta_step = domain_betas[loss_domain]
                            lambda_base = domain_lambdas[loss_domain]
                        else:
                            mean = physical_target_mean.view(1, 1, -1)
                            std = physical_target_std.view(1, 1, -1)
                            beta_step = phys_loss_beta
                            lambda_base = lambda_phys
                        for pred, phys in zip(phys_preds, physical_batches):
                            target = phys["target"]
                            valid = phys["valid"]
                            target = (target - mean) / std
                            phys_losses.append(
                                masked_smooth_l1_loss(
                                    pred,
                                    target,
                                    valid,
                                    beta=beta_step,
                                )
                            )
                            phys_maes.append(masked_l1_metric(pred.detach(), target, valid))
                        loss_phys = torch.stack(phys_losses).mean()
                        phys_mae_norm = torch.stack(phys_maes).mean()
                        global_step = epoch * ipe + itr
                        warmup_progress_step = global_step + 1
                        if (
                            time_jepa_multidomain
                            and domain_sampling_cfg["lambda_warmup_unit"]
                            == "domain_step"
                        ):
                            warmup_progress_step = domain_total_step_counts[
                                loss_domain
                            ]
                        if lambda_phys_warmup_steps > 0:
                            lambda_phys_step = lambda_base * min(
                                1.0,
                                float(warmup_progress_step)
                                / float(lambda_phys_warmup_steps),
                            )
                        else:
                            lambda_phys_step = lambda_base
                        loss = loss + loss_phys * lambda_phys_step

                # Step 2. Backward & step
                run_step = True
                if loss_reg_std_mult is not None:
                    meanval = np.mean(trailing_losses)
                    stdval = np.std(trailing_losses)
                    max_bound = meanval + loss_reg_std_mult * stdval
                    if (
                        loss > max_bound
                        and epoch > loss_reg_min_epoch
                        and len(trailing_losses)
                        > int(0.5 * loss_reg_num_tracking_steps)
                    ):
                        run_step = False
                        loss.backward()
                        logger.info(
                            f"Loss {loss} is above bound {meanval} + {loss_reg_std_mult} * {stdval}. Skipping step."
                        )

                if run_step:
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                m = min(next(momentum_scheduler), ema[1])
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(
                        encoder.parameters(), target_encoder.parameters()
                    ):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)

                return (
                    float(loss.detach()),
                    float(loss_jepa.detach()),
                    float(loss_phys.detach()),
                    float(phys_mae_norm.detach()),
                    float(lambda_phys_step),
                    _new_lr,
                    _new_wd,
                    run_step,
                )

            (
                loss,
                loss_jepa,
                loss_phys,
                phys_mae_norm,
                lambda_phys_step,
                _new_lr,
                _new_wd,
                run_step,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)
            if time_jepa_multidomain:
                domain_loss_meters[active_domain].update(loss)
                domain_jepa_loss_meters[active_domain].update(loss_jepa)
                if time_jepa_enabled:
                    domain_time_loss_meters[active_domain].update(loss_phys)
                    domain_time_mae_meters[active_domain].update(phys_mae_norm)

            if loss_reg_std_mult is not None:
                if run_step:
                    trailing_losses.append(loss)
                    if len(trailing_losses) > loss_reg_num_tracking_steps:
                        trailing_losses = trailing_losses[1:]
                else:
                    step_count += 1
                    if step_count > MAX_REPEAT_COUNTS:
                        raise RuntimeError(
                            "Loss is above bound for too many tries. Exiting."
                        )

            # -- Logging
            def log_stats():
                global_step = epoch * ipe + itr
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    iter_elapsed_time_ms,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", loss, global_step)
                    tb_writer.add_scalar("train/loss_avg", loss_meter.avg, global_step)
                    if time_jepa_enabled:
                        tb_writer.add_scalar("train/loss_jepa", loss_jepa, global_step)
                        tb_writer.add_scalar("train/loss_phys", loss_phys, global_step)
                        tb_writer.add_scalar("train/phys_mae_norm", phys_mae_norm, global_step)
                        tb_writer.add_scalar("train/lambda_phys", lambda_phys_step, global_step)
                    if time_jepa_multidomain:
                        domain_prefix = f"train_domain/{active_domain}"
                        tb_writer.add_scalar(f"{domain_prefix}/loss", loss, global_step)
                        tb_writer.add_scalar(
                            f"{domain_prefix}/loss_jepa",
                            loss_jepa,
                            global_step,
                        )
                        tb_writer.add_scalar(
                            f"{domain_prefix}/steps_epoch",
                            domain_step_counts[active_domain],
                            global_step,
                        )
                        tb_writer.add_scalar(
                            f"{domain_prefix}/steps_total",
                            domain_total_step_counts[active_domain],
                            global_step,
                        )
                        tb_writer.add_scalar(
                            f"{domain_prefix}/samples_epoch",
                            domain_sample_counts[active_domain],
                            global_step,
                        )
                        if time_jepa_enabled:
                            tb_writer.add_scalar(
                                f"{domain_prefix}/loss_time",
                                loss_phys,
                                global_step,
                            )
                            tb_writer.add_scalar(
                                f"{domain_prefix}/time_mae_norm",
                                phys_mae_norm,
                                global_step,
                            )
                    tb_writer.add_scalar("optim/lr", _new_lr, global_step)
                    tb_writer.add_scalar("optim/weight_decay", _new_wd, global_step)
                    tb_writer.add_scalar("time/iter_ms", iter_elapsed_time_ms, global_step)
                    tb_writer.add_scalar("time/gpu_ms", gpu_etime_ms, global_step)
                    tb_writer.add_scalar("time/dataload_ms", data_elapsed_time_ms, global_step)
                    if torch.cuda.is_available():
                        tb_writer.add_scalar(
                            "system/max_memory_mb",
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            global_step,
                        )
                    if global_step % log_freq == 0:
                        tb_writer.flush()
                if (
                    (itr % log_freq == 0)
                    or (itr == ipe - 1)
                    or np.isnan(loss)
                    or np.isinf(loss)
                ):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "%s"
                        "%s"
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            (
                                "jepa: %.3f phys: %.3f phys_mae_norm: %.3f lambda_phys: %.3f "
                                % (loss_jepa, loss_phys, phys_mae_norm, lambda_phys_step)
                                if time_jepa_enabled
                                else ""
                            ),
                            (
                                "domain: %s " % active_domain
                                if time_jepa_multidomain
                                else ""
                            ),
                            "["
                            + ", ".join(
                                [
                                    f"{k}: " + "%.1f" % mask_meters[k].avg
                                    for k in mask_meters
                                ]
                            )
                            + "]",
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            if pbar is not None:
                pbar.set_postfix(
                    loss=f"{loss_meter.avg:.3f}",
                    lr=f"{_new_lr:.2e}",
                    ms=f"{iter_time_meter.avg:.0f}",
                )
            assert not np.isnan(loss), "loss is nan"

        if pbar is not None:
            pbar.close()

        if time_jepa_multidomain:
            if (
                planned_domain_quotas is not None
                and domain_step_counts != planned_domain_quotas
            ):
                raise RuntimeError(
                    "Observed multi-domain step counts do not match the planned "
                    f"quota: observed={domain_step_counts}, planned={planned_domain_quotas}"
                )
            exposure_summary = {
                "epoch": epoch + 1,
                "sampling": domain_sampling_cfg,
                "planned_steps": planned_domain_quotas,
                "observed_steps": domain_step_counts,
                "observed_total_steps": domain_total_step_counts,
                "observed_global_samples": domain_sample_counts,
                "domains": {
                    name: {
                        "loss": domain_loss_meters[name].avg,
                        "loss_jepa": domain_jepa_loss_meters[name].avg,
                        **(
                            {
                                "loss_time": domain_time_loss_meters[name].avg,
                                "time_mae_norm": domain_time_mae_meters[name].avg,
                            }
                            if time_jepa_enabled
                            else {}
                        ),
                    }
                    for name in domain_names
                },
            }
            logger.info(
                "Epoch %d multi-domain exposure: %s",
                epoch + 1,
                exposure_summary,
            )
            if rank == 0:
                with open(domain_exposure_log, "a") as exposure_file:
                    exposure_file.write(
                        json.dumps(exposure_summary, sort_keys=True) + "\n"
                    )
                if tb_writer is not None:
                    for domain_name in domain_names:
                        epoch_prefix = f"train_epoch_domain/{domain_name}"
                        tb_writer.add_scalar(
                            f"{epoch_prefix}/steps",
                            domain_step_counts[domain_name],
                            epoch + 1,
                        )
                        tb_writer.add_scalar(
                            f"{epoch_prefix}/global_samples",
                            domain_sample_counts[domain_name],
                            epoch + 1,
                        )
                        tb_writer.add_scalar(
                            f"{epoch_prefix}/loss",
                            domain_loss_meters[domain_name].avg,
                            epoch + 1,
                        )
                        tb_writer.add_scalar(
                            f"{epoch_prefix}/loss_jepa",
                            domain_jepa_loss_meters[domain_name].avg,
                            epoch + 1,
                        )
                        if time_jepa_enabled:
                            tb_writer.add_scalar(
                                f"{epoch_prefix}/loss_time",
                                domain_time_loss_meters[domain_name].avg,
                                epoch + 1,
                            )
                            tb_writer.add_scalar(
                                f"{epoch_prefix}/time_mae_norm",
                                domain_time_mae_meters[domain_name].avg,
                                epoch + 1,
                            )

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_every_file = f"e{epoch}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
