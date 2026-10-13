"""Frozen iVideoGPT causal-world-state adapter for the physical probe.

This module is intentionally independent from the existing InternVideo
adapter.  It consumes the complete official Hugging Face snapshot (tokenizer
and transformer), keeps both components frozen, and exposes 32 causal states
for the established 64-frame / tubelet-2 physical-readout protocol.

The OXE checkpoint was trained on native 16-frame segments with two context
frames.  A 64-frame clip is therefore split into four non-overlapping native
windows instead of extrapolating the transformer to an untrained 64-frame
token sequence.  Within each window, the last-layer hidden state of the SDF
frame-boundary token after every pair of frames is used as the world state.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import torch

from .base import EncoderSpec, FeatureEncoderAdapter


REPO_ROOT = Path(__file__).resolve().parents[2]
IVIDEOGPT_ROOT = REPO_ROOT / "iVideoGPT"
HUGGINGFACE_REPO = "thuml/ivideogpt-oxe-64-act-free-medium"
PINNED_REVISION = "0076304ff7698d90d48ac6036022f5eff0ae2f3e"

REQUIRED_SNAPSHOT_FILES = (
    "tokenizer/config.json",
    "tokenizer/diffusion_pytorch_model.safetensors",
    "transformer/config.json",
    "transformer/model.safetensors",
)

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class IVideoGPTTokenLayout:
    """Exact token and state positions for one native iVideoGPT window."""

    window_frames: int
    context_frames: int
    target_group_size: int
    context_tokens_per_frame: int
    dynamics_tokens_per_frame: int
    context_sequence_tokens: int
    native_sequence_tokens: int
    sequence_tokens_with_terminal_sdf: int
    states_per_window: int
    boundary_indices: tuple

    def to_dict(self):
        value = asdict(self)
        value["boundary_indices"] = list(self.boundary_indices)
        return value


def build_token_layout(
    window_frames=16,
    context_frames=2,
    target_group_size=2,
    context_tokens_per_frame=256,
    dynamics_tokens_per_frame=16,
):
    """Build and validate the native token layout used by the adapter.

    ``tokenize`` removes the first context marker, so the native length is

        context_frames * (context_tokens + 1) - 1
        + future_frames * (dynamics_tokens + 1).

    One terminal SDF marker is appended without a prediction label.  Its
    hidden state summarizes the last observed frame pair in exactly the same
    marker format used at earlier frame boundaries.
    """

    values = {
        "window_frames": window_frames,
        "context_frames": context_frames,
        "target_group_size": target_group_size,
        "context_tokens_per_frame": context_tokens_per_frame,
        "dynamics_tokens_per_frame": dynamics_tokens_per_frame,
    }
    if any(int(value) <= 0 for value in values.values()):
        raise ValueError(f"Token-layout values must be positive: {values}")
    if window_frames <= context_frames:
        raise ValueError("window_frames must be greater than context_frames")
    if window_frames % target_group_size:
        raise ValueError(
            f"window_frames={window_frames} must be divisible by target_group_size={target_group_size}"
        )
    if context_frames != target_group_size:
        raise ValueError(
            "The registered physical protocol expects the first target pair to end "
            "at the two-frame context boundary"
        )

    context_sequence_tokens = context_frames * (context_tokens_per_frame + 1) - 1
    future_frames = window_frames - context_frames
    tokens_per_dynamics_frame = dynamics_tokens_per_frame + 1
    native_sequence_tokens = (
        context_sequence_tokens + future_frames * tokens_per_dynamics_frame
    )
    boundary_indices = tuple(
        context_sequence_tokens
        + (frame_end - context_frames) * tokens_per_dynamics_frame
        for frame_end in range(target_group_size, window_frames + 1, target_group_size)
    )
    if boundary_indices[-1] != native_sequence_tokens:
        raise RuntimeError(
            "The final physical state must use the appended terminal SDF marker: "
            f"last boundary={boundary_indices[-1]}, native length={native_sequence_tokens}"
        )

    return IVideoGPTTokenLayout(
        window_frames=window_frames,
        context_frames=context_frames,
        target_group_size=target_group_size,
        context_tokens_per_frame=context_tokens_per_frame,
        dynamics_tokens_per_frame=dynamics_tokens_per_frame,
        context_sequence_tokens=context_sequence_tokens,
        native_sequence_tokens=native_sequence_tokens,
        sequence_tokens_with_terminal_sdf=native_sequence_tokens + 1,
        states_per_window=window_frames // target_group_size,
        boundary_indices=boundary_indices,
    )


def resolve_snapshot_root(checkpoint_path):
    """Resolve and validate a complete local tokenizer+transformer snapshot."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.is_dir():
        snapshot_root = checkpoint_path
    elif checkpoint_path.name == "model.safetensors" and checkpoint_path.parent.name == "transformer":
        snapshot_root = checkpoint_path.parent.parent
    else:
        snapshot_root = checkpoint_path.parent

    missing = [
        relative_path
        for relative_path in REQUIRED_SNAPSHOT_FILES
        if not (snapshot_root / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete iVideoGPT checkpoint snapshot at {snapshot_root}; missing files: {missing}"
        )
    return snapshot_root


def split_native_windows(videos, window_frames):
    """Convert ``[B,C,T,H,W]`` into temporal-order ``[B*W,Tw,C,H,W]``."""

    if videos.ndim != 5:
        raise ValueError(f"Expected videos [B,C,T,H,W], got {tuple(videos.shape)}")
    batch_size, channels, total_frames, height, width = videos.shape
    if total_frames % window_frames:
        raise ValueError(
            f"total_frames={total_frames} must be divisible by window_frames={window_frames}"
        )
    num_windows = total_frames // window_frames
    return (
        videos.reshape(batch_size, channels, num_windows, window_frames, height, width)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(batch_size * num_windows, window_frames, channels, height, width)
        .contiguous()
    )


def extract_boundary_states(hidden_states, batch_size, num_windows, layout):
    """Select causal pair-end boundary states and restore full-clip order."""

    if hidden_states.ndim != 3:
        raise ValueError(
            f"Expected hidden states [B*W,L,D], got {tuple(hidden_states.shape)}"
        )
    expected_batch = batch_size * num_windows
    if hidden_states.shape[0] != expected_batch:
        raise ValueError(
            f"Hidden-state batch is {hidden_states.shape[0]}, expected {expected_batch}"
        )
    if hidden_states.shape[1] != layout.sequence_tokens_with_terminal_sdf:
        raise ValueError(
            f"Hidden sequence is {hidden_states.shape[1]}, expected "
            f"{layout.sequence_tokens_with_terminal_sdf}"
        )
    indices = torch.as_tensor(
        layout.boundary_indices,
        device=hidden_states.device,
        dtype=torch.long,
    )
    states = hidden_states.index_select(1, indices)
    return states.reshape(
        batch_size,
        num_windows * layout.states_per_window,
        hidden_states.shape[-1],
    )


def _read_json(path):
    with Path(path).open("r") as handle:
        return json.load(handle)


def _snapshot_revision(snapshot_root):
    metadata_path = (
        snapshot_root
        / ".cache/huggingface/download/transformer/model.safetensors.metadata"
    )
    if not metadata_path.is_file():
        return None
    lines = metadata_path.read_text().splitlines()
    return lines[0].strip() if lines else None


def _local_git_revision(repository_root):
    git_dir = Path(repository_root) / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text().strip()
    if not head.startswith("ref: "):
        return head or None
    ref_name = head.removeprefix("ref: ")
    ref_path = git_dir / ref_name
    if ref_path.is_file():
        return ref_path.read_text().strip() or None
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        suffix = f" {ref_name}"
        for line in packed_refs.read_text().splitlines():
            if line and not line.startswith(("#", "^")) and line.endswith(suffix):
                return line.split(" ", 1)[0]
    return None


def _json_compatible(value):
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


class IVideoGPTFrozenAdapter(FeatureEncoderAdapter):
    """Expose frozen iVideoGPT-OXE-64-M causal boundary states."""

    def __init__(
        self,
        checkpoint_path,
        input_frames=64,
        input_size=64,
        target_group_size=2,
        window_frames=16,
        context_frames=2,
        encoder_dtype="float32",
        attention_implementation="sdpa",
    ):
        super().__init__()
        if encoder_dtype not in DTYPES:
            raise ValueError(
                f"Unsupported encoder dtype {encoder_dtype!r}; choose from {tuple(DTYPES)}"
            )
        if attention_implementation not in ("sdpa", "eager"):
            raise ValueError("attention_implementation must be 'sdpa' or 'eager'")
        if input_frames % window_frames:
            raise ValueError(
                f"input_frames={input_frames} must be divisible by window_frames={window_frames}"
            )

        self.snapshot_root = resolve_snapshot_root(checkpoint_path)
        snapshot_revision = _snapshot_revision(self.snapshot_root)
        if snapshot_revision is not None and snapshot_revision != PINNED_REVISION:
            raise RuntimeError(
                "iVideoGPT snapshot revision does not match the registered baseline: "
                f"{snapshot_revision} != {PINNED_REVISION}"
            )
        tokenizer_config = _read_json(self.snapshot_root / "tokenizer/config.json")
        transformer_config = _read_json(self.snapshot_root / "transformer/config.json")

        actual_tokenizer = {
            "resolution": int(tokenizer_config.get("resolution", -1)),
            "context_length": int(tokenizer_config.get("context_length", -1)),
            "num_vq_embeddings": int(tokenizer_config.get("num_vq_embeddings", -1)),
            "num_dyn_embeddings": int(tokenizer_config.get("num_dyn_embeddings", -1)),
            "patch_size": int(tokenizer_config.get("patch_size", -1)),
        }
        expected_tokenizer = {
            "resolution": 64,
            "context_length": 2,
            "num_vq_embeddings": 8192,
            "num_dyn_embeddings": 8192,
            "patch_size": 4,
        }
        if actual_tokenizer != expected_tokenizer:
            raise RuntimeError(
                "This adapter is pinned to the OXE-64 action-free tokenizer; "
                f"expected {expected_tokenizer}, got {actual_tokenizer}"
            )

        actual_transformer = {
            "hidden_size": int(transformer_config.get("hidden_size", -1)),
            "num_hidden_layers": int(transformer_config.get("num_hidden_layers", -1)),
            "num_attention_heads": int(transformer_config.get("num_attention_heads", -1)),
            "vocab_size": int(transformer_config.get("vocab_size", -1)),
            "max_position_embeddings": int(
                transformer_config.get("max_position_embeddings", -1)
            ),
        }
        expected_transformer = {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "vocab_size": 16386,
            "max_position_embeddings": 1024,
        }
        if actual_transformer != expected_transformer:
            raise RuntimeError(
                "This adapter is pinned to the OXE-64 medium transformer; "
                f"expected {expected_transformer}, got {actual_transformer}"
            )
        if input_size != actual_tokenizer["resolution"]:
            raise ValueError(
                f"input_size must remain checkpoint-native 64, got {input_size}"
            )
        if context_frames != actual_tokenizer["context_length"]:
            raise ValueError(
                f"context_frames must remain checkpoint-native 2, got {context_frames}"
            )

        self.layout = build_token_layout(
            window_frames=window_frames,
            context_frames=context_frames,
            target_group_size=target_group_size,
        )
        if (
            self.layout.sequence_tokens_with_terminal_sdf
            > actual_transformer["max_position_embeddings"]
        ):
            raise RuntimeError(
                "Native window plus terminal marker exceeds the checkpoint context: "
                f"{self.layout.sequence_tokens_with_terminal_sdf} > "
                f"{actual_transformer['max_position_embeddings']}"
            )

        self.num_windows = input_frames // window_frames
        self.compute_dtype_name = encoder_dtype
        self.compute_dtype = DTYPES[encoder_dtype]
        self.attention_implementation = attention_implementation
        self.sdf_token_id = (
            actual_tokenizer["num_vq_embeddings"]
            + actual_tokenizer["num_dyn_embeddings"]
            + 1
        )

        # The shared probe interprets one adapter token as one tubelet-2 state.
        self.spec = EncoderSpec(
            name="ivideogpt_oxe_64_act_free_medium_causal_state",
            input_frames=input_frames,
            input_size=input_size,
            feature_dim=actual_transformer["hidden_size"],
            encoder_tubelet_size=target_group_size,
            spatial_tokens_per_frame=1,
            target_group_size=target_group_size,
        )

        if not IVIDEOGPT_ROOT.is_dir():
            raise FileNotFoundError(f"Local iVideoGPT source is missing: {IVIDEOGPT_ROOT}")
        if str(IVIDEOGPT_ROOT) not in sys.path:
            sys.path.insert(0, str(IVIDEOGPT_ROOT))

        try:
            from ivideogpt.vq_model import CompressiveVQModel
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the local iVideoGPT tokenizer. Install a compatible "
                "diffusers package in the isolated iVideoGPT environment."
            ) from exc

        from transformers import AutoModelForCausalLM

        tokenizer = CompressiveVQModel.from_pretrained(
            str(self.snapshot_root),
            subfolder="tokenizer",
            local_files_only=True,
            low_cpu_mem_usage=False,
        )
        tokenizer = tokenizer.to(dtype=self.compute_dtype)

        transformer_kwargs = {
            "subfolder": "transformer",
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "output_loading_info": True,
            "attn_implementation": attention_implementation,
        }
        # Transformers 5 renamed torch_dtype to dtype.  Select the keyword by
        # version so the adapter remains usable with the official 4.38 stack.
        import diffusers
        import transformers

        major_version = int(transformers.__version__.split(".", 1)[0])
        transformer_kwargs["dtype" if major_version >= 5 else "torch_dtype"] = self.compute_dtype
        loaded = AutoModelForCausalLM.from_pretrained(
            str(self.snapshot_root),
            **transformer_kwargs,
        )
        if isinstance(loaded, tuple):
            transformer, loading_info = loaded
        else:
            transformer, loading_info = loaded, {}
        if not hasattr(transformer, "model"):
            raise RuntimeError(
                f"Unexpected transformer structure: {type(transformer).__name__} has no .model"
            )
        if transformer.config.vocab_size != self.sdf_token_id + 1:
            raise RuntimeError(
                f"Transformer vocab={transformer.config.vocab_size}, expected {self.sdf_token_id + 1}"
            )
        transformer.config.use_cache = False

        self.tokenizer = tokenizer.eval()
        self.transformer = transformer.eval()
        self.register_buffer(
            "_boundary_indices",
            torch.tensor(self.layout.boundary_indices, dtype=torch.long),
            persistent=False,
        )

        spec_report = self.spec.to_dict()
        spec_report["output_layout"] = "[batch, 32 causal pair-end boundary states, feature]"
        spec_report["encoder_tubelet_size_note"] = (
            "Adapter state stride only; iVideoGPT uses discrete frame tokens, not video tubelets"
        )
        self.load_report = {
            "source": "local_huggingface_snapshot",
            "local_source_root": str(IVIDEOGPT_ROOT),
            "local_source_revision": _local_git_revision(IVIDEOGPT_ROOT),
            "huggingface_repo": HUGGINGFACE_REPO,
            "pinned_revision": PINNED_REVISION,
            "snapshot_revision": snapshot_revision,
            "snapshot_root": str(self.snapshot_root),
            "files": {
                relative_path: (self.snapshot_root / relative_path).stat().st_size
                for relative_path in REQUIRED_SNAPSHOT_FILES
            },
            "dtype": encoder_dtype,
            "attention_implementation": attention_implementation,
            "software_versions": {
                "torch": str(torch.__version__),
                "transformers": transformers.__version__,
                "diffusers": diffusers.__version__,
            },
            "input_range": "[0, 1] without ImageNet normalization",
            "temporal_protocol": {
                "full_clip_frames": input_frames,
                "native_window_frames": window_frames,
                "num_non_overlapping_windows": self.num_windows,
                "cross_window_memory": False,
                "causal_within_window": True,
                "future_frame_leakage": False,
            },
            "token_layout": self.layout.to_dict(),
            "state_extraction": {
                "source": "transformer.model last_hidden_state",
                "pooling": "SDF frame-boundary state after every two observed frames",
                "terminal_sdf_appended": True,
                "lm_head_computed": False,
                "use_cache": False,
            },
            "tokenizer_config": actual_tokenizer,
            "transformer_config": actual_transformer,
            "loading_info": {
                key: _json_compatible(value)
                for key, value in loading_info.items()
                if key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            },
            "parameter_count": {
                "tokenizer": sum(parameter.numel() for parameter in self.tokenizer.parameters()),
                "transformer_with_lm_head": sum(
                    parameter.numel() for parameter in self.transformer.parameters()
                ),
            },
            "spec": spec_report,
        }

    def _autocast_context(self, device_type):
        if device_type not in ("cpu", "cuda"):
            return nullcontext()
        if self.compute_dtype == torch.float32:
            # Override the shared probe runner's outer FP16 autocast.  The
            # official checkpoint and tokenizer quantization are FP32.
            return torch.amp.autocast(device_type=device_type, enabled=False)
        return torch.amp.autocast(
            device_type=device_type,
            enabled=True,
            dtype=self.compute_dtype,
        )

    def forward(self, videos):
        if videos.ndim != 5:
            raise ValueError(f"Expected video [B,C,T,H,W], got {tuple(videos.shape)}")
        batch_size, channels, frames, height, width = videos.shape
        expected = (3, self.spec.input_frames, self.spec.input_size, self.spec.input_size)
        if (channels, frames, height, width) != expected:
            raise ValueError(
                "iVideoGPT input mismatch: "
                f"got C/T/H/W={(channels, frames, height, width)}, expected {expected}"
            )

        pixel_values = split_native_windows(videos, self.layout.window_frames)
        pixel_values = pixel_values.to(dtype=self.compute_dtype)
        with self._autocast_context(videos.device.type):
            tokens, _ = self.tokenizer.tokenize(
                pixel_values,
                context_length=self.layout.context_frames,
            )
            if tokens.shape[1] != self.layout.native_sequence_tokens:
                raise RuntimeError(
                    f"Tokenizer returned {tokens.shape[1]} tokens per window, expected "
                    f"{self.layout.native_sequence_tokens}"
                )

            existing_boundaries = self._boundary_indices[:-1]
            if not torch.all(tokens.index_select(1, existing_boundaries) == self.sdf_token_id):
                raise RuntimeError(
                    "Tokenizer frame-boundary markers do not match the registered SDF layout"
                )
            terminal_sdf = torch.full(
                (tokens.shape[0], 1),
                self.sdf_token_id,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens = torch.cat((tokens, terminal_sdf), dim=1)

            outputs = self.transformer.model(
                input_ids=tokens,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            features = extract_boundary_states(
                hidden_states,
                batch_size=batch_size,
                num_windows=self.num_windows,
                layout=self.layout,
            )

        expected_shape = (batch_size, self.spec.output_tokens, self.spec.feature_dim)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"iVideoGPT features have shape {tuple(features.shape)}, expected {expected_shape}"
            )
        return features
