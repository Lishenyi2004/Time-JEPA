"""VideoMAE V2 ViT-g encoder adapter for the frozen physical probe.

Only the clean-pretraining encoder is loaded.  The MAE decoder, mask token and
encoder-to-decoder projection in the public checkpoint are intentionally
discarded.  The adapter exposes all patch tokens in temporal-major order.
"""

from contextlib import contextmanager
from functools import partial
from pathlib import Path
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EncoderSpec, FeatureEncoderAdapter


OFFICIAL_CHECKPOINT_BYTES = 4_105_433_973
OFFICIAL_CHECKPOINT_SHA256 = (
    "5773c881d303426f2f0e906df9842c4ca95fba6a5901849a3e8d9d1ed8f79b5f"
)
EXPECTED_ENCODER_TENSORS = 524
EXPECTED_ENCODER_PARAMETERS = 1_011_610_752

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_sinusoid_position_embedding(n_position, embed_dim, dtype=torch.float32):
    """Vectorized equivalent of VideoMAE V2's fixed 1-D sinusoid table."""

    if n_position <= 0 or embed_dim <= 0:
        raise ValueError(
            f"n_position and embed_dim must be positive, got {n_position}, {embed_dim}"
        )
    positions = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(embed_dim, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000.0) * (2.0 * torch.div(dimensions, 2, rounding_mode="floor"))
        / embed_dim
    )
    angles = positions * frequencies.unsqueeze(0)
    table = torch.empty_like(angles)
    table[:, 0::2] = torch.sin(angles[:, 0::2])
    table[:, 1::2] = torch.cos(angles[:, 1::2])
    return table.unsqueeze(0).to(dtype=dtype)


def extract_encoder_state(
    checkpoint,
    expected_tensor_count=EXPECTED_ENCODER_TENSORS,
    expected_parameter_count=EXPECTED_ENCODER_PARAMETERS,
):
    """Select ``model['encoder.*']`` and remove the encoder prefix.

    Optional expected counts make the helper usable in small unit tests while
    keeping strict validation enabled for the official checkpoint loader.
    """

    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("VideoMAE V2 checkpoint must contain a mapping at key 'model'")
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in checkpoint["model"].items()
        if isinstance(key, str) and key.startswith("encoder.")
    }
    non_tensors = sorted(
        key for key, value in encoder_state.items() if not isinstance(value, torch.Tensor)
    )
    if non_tensors:
        raise RuntimeError(f"Non-tensor entries in VideoMAE encoder state: {non_tensors}")

    tensor_count = len(encoder_state)
    parameter_count = sum(value.numel() for value in encoder_state.values())
    if expected_tensor_count is not None and tensor_count != expected_tensor_count:
        raise RuntimeError(
            "VideoMAE V2-g encoder tensor count mismatch: "
            f"got {tensor_count}, expected {expected_tensor_count}"
        )
    if expected_parameter_count is not None and parameter_count != expected_parameter_count:
        raise RuntimeError(
            "VideoMAE V2-g encoder parameter count mismatch: "
            f"got {parameter_count:,}, expected {expected_parameter_count:,}"
        )
    return encoder_state


def _fused_sdpa(query, key, value, dropout_p, scale):
    """Use fused CUDA SDPA backends and refuse an O(N^2) math fallback."""

    kwargs = {
        "dropout_p": dropout_p,
        "is_causal": False,
        "scale": scale,
    }
    if not query.is_cuda:
        return F.scaled_dot_product_attention(query, key, value, **kwargs)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:  # pragma: no cover - current supported env is PyTorch 2.9
        raise RuntimeError(
            "VideoMAE V2 64-frame extraction requires PyTorch fused SDPA support"
        ) from exc

    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    if hasattr(SDPBackend, "CUDNN_ATTENTION"):
        backends.append(SDPBackend.CUDNN_ATTENTION)
    with sdpa_kernel(backends=backends):
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


def _sdpa_attention_forward(attention, x):
    """VideoMAE ``Attention.forward`` rewritten with equivalent fused SDPA."""

    batch_size, num_tokens, _ = x.shape
    qkv_bias = None
    if attention.q_bias is not None:
        qkv_bias = torch.cat(
            (
                attention.q_bias,
                torch.zeros_like(attention.v_bias, requires_grad=False),
                attention.v_bias,
            )
        )
    qkv = F.linear(x, weight=attention.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(
        batch_size, num_tokens, 3, attention.num_heads, -1
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]
    dropout_p = attention.attn_drop.p if attention.training else 0.0
    output = _fused_sdpa(
        query,
        key,
        value,
        dropout_p=dropout_p,
        scale=attention.scale,
    )
    output = output.transpose(1, 2).reshape(batch_size, num_tokens, -1)
    output = attention.proj(output)
    return attention.proj_drop(output)


def enable_sdpa_attention(encoder):
    """Patch every attention instance without changing checkpoint key names."""

    patched = 0
    for block in encoder.blocks:
        block.attn.forward = types.MethodType(_sdpa_attention_forward, block.attn)
        block.attn._world_model_attention_implementation = "fused_sdpa"
        patched += 1
    return patched


@contextmanager
def _cheap_meta_constructor(modeling_pretrain):
    """Build the 1B module on meta without running costly random initialization.

    The upstream constructor calls ``Tensor.item`` on a linspace, which cannot
    run on meta.  That one tiny schedule is therefore produced on CPU.  All
    monkey patches are process-local and restored before checkpoint loading.
    """

    encoder_class = modeling_pretrain.PretrainVisionTransformerEncoder
    original_init_weights = encoder_class._init_weights
    original_position_builder = modeling_pretrain.get_sinusoid_encoding_table
    original_linspace = torch.linspace

    def no_init(self, module):
        del self, module

    def empty_position_table(n_position, embed_dim):
        del n_position
        return torch.empty((1, 0, embed_dim), dtype=torch.float32, device="cpu")

    def cpu_linspace(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)

    encoder_class._init_weights = no_init
    modeling_pretrain.get_sinusoid_encoding_table = empty_position_table
    torch.linspace = cpu_linspace
    try:
        yield
    finally:
        torch.linspace = original_linspace
        modeling_pretrain.get_sinusoid_encoding_table = original_position_builder
        encoder_class._init_weights = original_init_weights


def _build_meta_giant_encoder(model_frames):
    try:
        from VideoMAEv2.models import modeling_pretrain
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the local VideoMAEv2 source tree. Use the existing "
            "jepa or internvideo_probe environment; do not install the repository's "
            "historical pinned requirements over it."
        ) from exc

    with _cheap_meta_constructor(modeling_pretrain):
        with torch.device("meta"):
            encoder = modeling_pretrain.PretrainVisionTransformerEncoder(
                img_size=224,
                patch_size=14,
                in_chans=3,
                num_classes=0,
                embed_dim=1408,
                depth=40,
                num_heads=16,
                mlp_ratio=48 / 11,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.0,
                tubelet_size=2,
                use_learnable_pos_emb=False,
                with_cp=False,
                all_frames=model_frames,
                cos_attn=False,
            )
    return encoder


def _load_official_encoder(checkpoint_path, model_frames, encoder_dtype):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VideoMAE V2 checkpoint not found: {checkpoint_path}")
    file_bytes = checkpoint_path.stat().st_size
    if file_bytes != OFFICIAL_CHECKPOINT_BYTES:
        raise RuntimeError(
            "VideoMAE V2-g checkpoint is incomplete or is not the pinned clean "
            f"checkpoint: got {file_bytes:,} bytes, expected {OFFICIAL_CHECKPOINT_BYTES:,}"
        )

    encoder = _build_meta_giant_encoder(model_frames=model_frames)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        mmap_used = True
    except TypeError:  # pragma: no cover - compatibility with older torch only
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        mmap_used = False

    encoder_state = extract_encoder_state(checkpoint)
    expected_keys = set(encoder.state_dict())
    supplied_keys = set(encoder_state)
    if expected_keys != supplied_keys:
        missing = sorted(expected_keys - supplied_keys)
        unexpected = sorted(supplied_keys - expected_keys)
        raise RuntimeError(
            "VideoMAE V2-g encoder key mismatch before loading: "
            f"missing={missing}, unexpected={unexpected}"
        )

    dtype = DTYPES[encoder_dtype]
    cast_state = {
        key: value.to(dtype=dtype) if value.is_floating_point() else value
        for key, value in encoder_state.items()
    }
    encoder.load_state_dict(cast_state, strict=True, assign=True)
    unresolved_meta = [name for name, parameter in encoder.named_parameters() if parameter.is_meta]
    if unresolved_meta:
        raise RuntimeError(
            f"VideoMAE V2-g parameters remained on meta after loading: {unresolved_meta}"
        )
    wrong_dtype = [
        name
        for name, parameter in encoder.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    ]
    if wrong_dtype:
        raise RuntimeError(
            f"VideoMAE V2-g parameters were not converted to {dtype}: {wrong_dtype}"
        )

    if hasattr(encoder, "pos_embed"):
        delattr(encoder, "pos_embed")
    position_table = build_sinusoid_position_embedding(
        encoder.patch_embed.num_patches,
        encoder.embed_dim,
        dtype=dtype,
    )
    encoder.register_buffer("pos_embed", position_table, persistent=False)

    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_bytes": file_bytes,
        "checkpoint_expected_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "checkpoint_sha256_verified_at_runtime": False,
        "torch_load_mmap": mmap_used,
        "checkpoint_top_level_keys": sorted(str(key) for key in checkpoint),
        "encoder_state_tensors": len(encoder_state),
        "encoder_parameter_count": sum(value.numel() for value in encoder_state.values()),
        "model_frames": model_frames,
        "position_embedding_shape": list(position_table.shape),
        "position_embedding_registered_nonpersistent_buffer": True,
    }
    return encoder, report


class VideoMAEV2GAdapter(FeatureEncoderAdapter):
    """Expose clean-pretrained VideoMAE V2-g patch tokens to the shared probe."""

    def __init__(
        self,
        checkpoint_path,
        input_frames=64,
        input_size=224,
        target_group_size=2,
        encoder_dtype="float16",
        attention_implementation="sdpa",
        temporal_mode="full",
        encoder_microbatch_size=1,
    ):
        super().__init__()
        if encoder_dtype not in DTYPES:
            raise ValueError(
                f"Unsupported encoder dtype {encoder_dtype!r}; choose from {tuple(DTYPES)}"
            )
        if input_frames != 64 or input_size != 224:
            raise ValueError(
                "The aligned VideoMAE V2-g protocol is pinned to 64x224 input, "
                f"got frames={input_frames}, size={input_size}"
            )
        if target_group_size != 2:
            raise ValueError(
                "VideoMAE V2-g tubelets and physical targets are both pinned to size 2"
            )
        if temporal_mode not in ("full", "native_chunks"):
            raise ValueError("temporal_mode must be one of: full, native_chunks")
        if attention_implementation not in ("sdpa", "eager"):
            raise ValueError("attention_implementation must be one of: sdpa, eager")
        if temporal_mode == "full" and attention_implementation == "eager":
            raise ValueError(
                "eager attention is unsafe for 8,192 tokens; use attention_implementation=sdpa"
            )
        if encoder_microbatch_size <= 0:
            raise ValueError("encoder_microbatch_size must be positive")

        self.spec = EncoderSpec(
            name="videomaev2_giant_p14_res224_clean_pretrain",
            input_frames=input_frames,
            input_size=input_size,
            feature_dim=1408,
            encoder_tubelet_size=2,
            spatial_tokens_per_frame=16 * 16,
            target_group_size=target_group_size,
        )
        self.temporal_mode = temporal_mode
        self.encoder_microbatch_size = encoder_microbatch_size
        model_frames = input_frames if temporal_mode == "full" else 16
        self.encoder, checkpoint_report = _load_official_encoder(
            checkpoint_path=checkpoint_path,
            model_frames=model_frames,
            encoder_dtype=encoder_dtype,
        )
        patched_blocks = 0
        if attention_implementation == "sdpa":
            patched_blocks = enable_sdpa_attention(self.encoder)
            if patched_blocks != 40:
                raise RuntimeError(
                    f"Expected to patch 40 VideoMAE attention blocks, got {patched_blocks}"
                )

        self.load_report = {
            "source": "OpenGVLab/VideoMAE2",
            "checkpoint_revision": "706cc172d65ebd4dedbee3f9c0183a93df9fa125",
            "checkpoint_variant": "mae-g/vit_g_hybrid_pt_1200e.pth",
            "checkpoint_training": "clean_self_supervised_pretraining_no_classification_finetune",
            "loaded_components": ["encoder"],
            "discarded_components": ["mask_token", "encoder_to_decoder", "decoder"],
            "dtype": encoder_dtype,
            "attention_implementation": attention_implementation,
            "attention_blocks_patched": patched_blocks,
            "cuda_math_sdpa_fallback_allowed": False,
            "temporal_mode": temporal_mode,
            "temporal_attention_scope": (
                "joint_64_frames"
                if temporal_mode == "full"
                else "four_independent_native_16_frame_chunks"
            ),
            "encoder_microbatch_size": encoder_microbatch_size,
            "checkpoint": checkpoint_report,
            "spec": self.spec.to_dict(),
        }

    def _encode_microbatched(self, videos):
        outputs = []
        num_patches = self.encoder.patch_embed.num_patches
        for video_batch in videos.split(self.encoder_microbatch_size, dim=0):
            mask = torch.zeros(
                (video_batch.shape[0], num_patches),
                dtype=torch.bool,
                device=video_batch.device,
            )
            outputs.append(self.encoder(video_batch, mask))
        return torch.cat(outputs, dim=0)

    def forward(self, videos):
        if videos.ndim != 5:
            raise ValueError(f"Expected video [B, C, T, H, W], got {tuple(videos.shape)}")
        _, channels, frames, height, width = videos.shape
        expected = (3, self.spec.input_frames, self.spec.input_size, self.spec.input_size)
        if (channels, frames, height, width) != expected:
            raise ValueError(
                "VideoMAE V2-g input mismatch: "
                f"got C/T/H/W={(channels, frames, height, width)}, expected {expected}"
            )

        if self.temporal_mode == "full":
            features = self._encode_microbatched(videos)
        else:
            features = torch.cat(
                [
                    self._encode_microbatched(videos[:, :, start : start + 16])
                    for start in range(0, self.spec.input_frames, 16)
                ],
                dim=1,
            )

        expected_shape = (videos.shape[0], self.spec.output_tokens, self.spec.feature_dim)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"VideoMAE V2-g features have shape {tuple(features.shape)}, "
                f"expected {expected_shape}"
            )
        return features
