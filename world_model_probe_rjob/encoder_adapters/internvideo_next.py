"""InternVideo-Next-L Hugging Face checkpoint adapter.

The public checkpoint is native to 16 frames.  For the aligned 64-frame
physical-readout protocol, this module expands the learned joint position
embedding once on CPU and replaces the native parameter before the model is
moved to a GPU.  Consequently, the upstream ``expand_pos_embed`` function is
never called in the per-batch forward path.
"""

from contextlib import contextmanager, nullcontext
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EncoderSpec, FeatureEncoderAdapter


REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "model.safetensors",
    "modeling_config.py",
    "modeling_internvideo_next.py",
    "preprocessor_config.json",
)

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _json_compatible(value):
    """Normalize Transformers loading metadata for logs and checkpoints."""

    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_compatible(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def resolve_snapshot_dir(checkpoint_path):
    """Resolve and validate a local Hugging Face checkpoint snapshot."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    snapshot_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    missing = [name for name in REQUIRED_SNAPSHOT_FILES if not (snapshot_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete InternVideo-Next checkpoint snapshot at {snapshot_dir}; "
            f"missing files: {missing}"
        )
    return snapshot_dir


def _snapshot_revision(snapshot_dir):
    metadata_path = (
        snapshot_dir
        / ".cache/huggingface/download/model.safetensors.metadata"
    )
    if not metadata_path.is_file():
        return None
    lines = metadata_path.read_text().splitlines()
    return lines[0].strip() if lines else None


def interpolate_temporal_position_embedding(
    pos_embed,
    source_frames,
    target_frames,
    spatial_tokens_per_frame,
    cls_tokens=4,
):
    """Linearly interpolate only the temporal axis of a joint position table.

    The spatial patch index and embedding channel are independent interpolation
    lanes.  Class-token positions are copied without modification.  Computation
    is deliberately performed in float32 on CPU and cast back to the source
    dtype, making this suitable as a one-time initialization step.
    """

    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f"Expected position embedding [1, N, D], got {tuple(pos_embed.shape)}")
    expected_tokens = cls_tokens + source_frames * spatial_tokens_per_frame
    if pos_embed.shape[1] != expected_tokens:
        raise ValueError(
            "Native position embedding does not match the declared token grid: "
            f"shape={tuple(pos_embed.shape)}, expected_tokens={expected_tokens}"
        )
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if target_frames == source_frames:
        return pos_embed.detach().clone()

    source_device = pos_embed.device
    source_dtype = pos_embed.dtype
    extra_tokens = pos_embed[:, :cls_tokens].detach().to(device="cpu", dtype=torch.float32)
    patch_tokens = pos_embed[:, cls_tokens:].detach().to(device="cpu", dtype=torch.float32)
    embed_dim = patch_tokens.shape[-1]

    # [1, T, S, D] -> [S, D, T], interpolate T, then restore temporal-major order.
    patch_tokens = patch_tokens.reshape(
        1, source_frames, spatial_tokens_per_frame, embed_dim
    )
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(
        spatial_tokens_per_frame, embed_dim, source_frames
    )
    patch_tokens = F.interpolate(
        patch_tokens,
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    patch_tokens = patch_tokens.reshape(
        1, spatial_tokens_per_frame, embed_dim, target_frames
    )
    patch_tokens = patch_tokens.permute(0, 3, 1, 2).reshape(
        1, target_frames * spatial_tokens_per_frame, embed_dim
    )
    expanded = torch.cat((extra_tokens, patch_tokens), dim=1)
    return expanded.to(device=source_device, dtype=source_dtype)


class _PortableDropoutAddRMSNorm(nn.Module):
    """Mathematical fallback for an optional FlashAttention fused RMSNorm op."""

    def __init__(
        self,
        hidden_size,
        prenorm=False,
        p=0.0,
        eps=1e-5,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x0, residual=None):
        if self.training and self.p:
            x0 = F.dropout(x0, p=self.p, training=True)
        residual_out = x0 if residual is None else x0 + residual
        if residual is None and self.residual_in_fp32:
            residual_out = residual_out.float()
        input_dtype = residual_out.dtype
        normalized = residual_out.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        normalized = normalized.to(input_dtype) * self.weight
        return (normalized, residual_out) if self.prenorm else normalized


@contextmanager
def _temporary_rms_norm_import_shim():
    """Let the upstream custom module import when optional fused norm is absent."""

    module_name = "flash_attn.ops.rms_norm"
    previous = sys.modules.get(module_name)
    shim = types.ModuleType(module_name)
    shim.DropoutAddRMSNorm = _PortableDropoutAddRMSNorm
    shim.RMSNorm = _PortableDropoutAddRMSNorm
    sys.modules[module_name] = shim
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def inspect_flashattention_kernels():
    """Return availability of the main attention and optional fused kernels."""

    report = {
        "flash_attention": False,
        "dropout_layer_norm": importlib.util.find_spec("dropout_layer_norm") is not None,
        "fused_dense_lib": importlib.util.find_spec("fused_dense_lib") is not None,
    }
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

        report["flash_attention"] = flash_attn_varlen_qkvpacked_func is not None
    except Exception as exc:  # pragma: no cover - depends on the CUDA environment
        report["flash_attention_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from flash_attn.modules.mlp import FusedMLP

        report["fused_mlp_class"] = FusedMLP is not None
    except Exception as exc:  # pragma: no cover - depends on the CUDA environment
        report["fused_mlp_class"] = False
        report["fused_mlp_error"] = f"{type(exc).__name__}: {exc}"
    report["native_fused_stack"] = all(
        report.get(key, False)
        for key in ("flash_attention", "dropout_layer_norm", "fused_dense_lib", "fused_mlp_class")
    )
    return report


def _enable_flash_attention_on_portable_backbone(backbone):
    """Use FlashAttention with ordinary RMSNorm/MLP blocks.

    The upstream implementation unnecessarily ties all three implementation
    choices in one assertion.  Building the mathematically equivalent ordinary
    RMSNorm/MLP blocks first and enabling only its memory-efficient attention
    path avoids the optional fused-dense and fused-layernorm extensions while
    retaining the 64-frame feasible attention kernel.
    """

    patched = 0
    for block in backbone.blocks:
        attention = block.attn
        custom_module = importlib.import_module(attention.__class__.__module__)
        attention.inner_attn = custom_module.FlashAttention(
            attention_dropout=attention.attn_drop.p
        )
        attention.causal = False
        attention.use_flash_attn = True
        patched += 1
    return patched


class InternVideoNextAdapter(FeatureEncoderAdapter):
    """Expose raw InternVideo-Next-L patch tokens to the shared physical probe."""

    def __init__(
        self,
        checkpoint_path,
        input_frames=64,
        input_size=224,
        target_group_size=2,
        encoder_dtype="float16",
        kernel_mode="auto",
    ):
        super().__init__()
        if encoder_dtype not in DTYPES:
            raise ValueError(f"Unsupported encoder dtype {encoder_dtype!r}; choose from {tuple(DTYPES)}")
        if kernel_mode not in ("auto", "native", "portable"):
            raise ValueError("kernel_mode must be one of: auto, native, portable")

        self.snapshot_dir = resolve_snapshot_dir(checkpoint_path)
        raw_config = json.loads((self.snapshot_dir / "config.json").read_text())
        model_config = raw_config.get("model_config", {})
        native_frames = int(model_config.get("num_frames", -1))
        native_size = int(model_config.get("img_size", -1))
        patch_size = int(model_config.get("patch_size", -1))
        tubelet_size = int(model_config.get("tubelet_size", -1))
        embed_dim = int(model_config.get("embed_dim", -1))
        cls_tokens = 4

        if (native_frames, native_size, patch_size, tubelet_size, embed_dim) != (
            16,
            224,
            14,
            1,
            1024,
        ):
            raise RuntimeError(
                "This adapter is pinned to InternVideo-Next-L p14/res224/f16, but "
                f"config reports frames={native_frames}, size={native_size}, "
                f"patch={patch_size}, tubelet={tubelet_size}, dim={embed_dim}"
            )
        if input_size != native_size:
            raise ValueError(
                f"InternVideo-Next-L input_size must remain checkpoint-native {native_size}, got {input_size}"
            )
        if input_frames % target_group_size:
            raise ValueError(
                f"input_frames={input_frames} must be divisible by target_group_size={target_group_size}"
            )

        spatial_grid = input_size // patch_size
        spatial_tokens = spatial_grid * spatial_grid
        self.spec = EncoderSpec(
            name="internvideo_next_large_p14_res224_f16",
            input_frames=input_frames,
            input_size=input_size,
            feature_dim=embed_dim,
            encoder_tubelet_size=tubelet_size,
            spatial_tokens_per_frame=spatial_tokens,
            target_group_size=target_group_size,
        )

        kernel_report = inspect_flashattention_kernels()
        if not kernel_report.get("flash_attention"):
            raise RuntimeError(
                "The main FlashAttention CUDA extension is unavailable.  T=64 cannot "
                f"safely fall back to quadratic naive attention. Details: {kernel_report}"
            )
        effective_kernel_mode = kernel_mode
        if kernel_mode == "auto":
            effective_kernel_mode = (
                "native" if kernel_report["native_fused_stack"] else "portable"
            )
        if effective_kernel_mode == "native" and not kernel_report["native_fused_stack"]:
            raise RuntimeError(
                "kernel_mode=native requires dropout_layer_norm and fused_dense_lib in "
                f"addition to FlashAttention. Details: {kernel_report}"
            )

        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(
            self.snapshot_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        config.model_config = dict(config.model_config)
        use_native_fused = effective_kernel_mode == "native"
        config.model_config.update(
            {
                "use_flash_attn": use_native_fused,
                "use_fused_rmsnorm": use_native_fused,
                "use_fused_mlp": use_native_fused,
            }
        )

        import_context = (
            nullcontext() if use_native_fused else _temporary_rms_norm_import_shim()
        )
        with import_context:
            loaded = AutoModel.from_pretrained(
                self.snapshot_dir,
                config=config,
                trust_remote_code=True,
                local_files_only=True,
                dtype=DTYPES[encoder_dtype],
                output_loading_info=True,
            )
        if isinstance(loaded, tuple):
            model, loading_info = loaded
        else:  # Defensive compatibility with Transformers versions ignoring the flag.
            model, loading_info = loaded, {}

        backbone = getattr(model, "model", None)
        if backbone is None or not hasattr(backbone, "patch_embed"):
            raise RuntimeError(
                f"Unexpected Hugging Face model structure: {type(model).__name__} has no .model.patch_embed"
            )
        patched_attention_blocks = 0
        if effective_kernel_mode == "portable":
            patched_attention_blocks = _enable_flash_attention_on_portable_backbone(backbone)

        native_pos_shape = tuple(backbone.pos_embed.shape)
        expanded_pos = interpolate_temporal_position_embedding(
            backbone.pos_embed,
            source_frames=native_frames,
            target_frames=input_frames,
            spatial_tokens_per_frame=spatial_tokens,
            cls_tokens=cls_tokens,
        )
        backbone.pos_embed = nn.Parameter(
            expanded_pos,
            requires_grad=backbone.pos_embed.requires_grad,
        )
        # Keep metadata consistent with the cached table. PatchEmbed.forward uses
        # the actual input shape, but these fields are useful to downstream code.
        backbone.num_frames = input_frames
        backbone.patch_embed.grid_size = (input_frames, spatial_grid, spatial_grid)
        backbone.patch_embed.num_patches = input_frames * spatial_tokens

        expected_pos_tokens = cls_tokens + self.spec.output_tokens
        if backbone.pos_embed.shape[1] != expected_pos_tokens:
            raise RuntimeError(
                f"Cached position table has {backbone.pos_embed.shape[1]} tokens; "
                f"expected {expected_pos_tokens}"
            )

        self.encoder = model
        self.load_report = {
            "source": "local_huggingface_snapshot",
            "huggingface_repo": "revliter/internvideo_next_large_p14_res224_f16",
            "snapshot_revision": _snapshot_revision(self.snapshot_dir),
            "snapshot_dir": str(self.snapshot_dir),
            "model_file_bytes": (self.snapshot_dir / "model.safetensors").stat().st_size,
            "kernel_mode_requested": kernel_mode,
            "kernel_mode_effective": effective_kernel_mode,
            "kernel_support": kernel_report,
            "portable_attention_blocks_patched": patched_attention_blocks,
            "dtype": encoder_dtype,
            "position_embedding": {
                "method": "cached_linear_temporal_interpolation_align_corners_false",
                "source_frames": native_frames,
                "target_frames": input_frames,
                "native_shape": list(native_pos_shape),
                "cached_shape": list(backbone.pos_embed.shape),
                "per_batch_interpolation": False,
            },
            "loading_info": {
                key: _json_compatible(value)
                for key, value in loading_info.items()
                if key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            },
            "spec": self.spec.to_dict(),
        }

    @property
    def backbone(self):
        return self.encoder.model

    def forward(self, videos):
        if videos.ndim != 5:
            raise ValueError(f"Expected video [B, C, T, H, W], got {tuple(videos.shape)}")
        _, channels, frames, height, width = videos.shape
        expected = (3, self.spec.input_frames, self.spec.input_size, self.spec.input_size)
        if (channels, frames, height, width) != expected:
            raise ValueError(
                "InternVideo-Next input mismatch: "
                f"got C/T/H/W={(channels, frames, height, width)}, expected {expected}"
            )
        expected_pos_tokens = 4 + self.spec.output_tokens
        if self.backbone.pos_embed.shape[1] != expected_pos_tokens:
            raise RuntimeError(
                "Position cache was changed after initialization; refusing to enter the "
                "upstream per-batch CPU interpolation path"
            )

        features = self.encoder.extract_features(pixel_values=videos)
        expected_shape = (videos.shape[0], self.spec.output_tokens, self.spec.feature_dim)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"InternVideo-Next features have shape {tuple(features.shape)}, expected {expected_shape}"
            )
        return features
