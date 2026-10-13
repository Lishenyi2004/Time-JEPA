"""Frozen feature adapter for a formally adapted iVideoGPT transformer."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch

from .ivideogpt import IVideoGPTFrozenAdapter, _json_compatible


def resolve_adapted_transformer(path):
    # Keep the checkpoint's lexical parent so a symlinked weights file still
    # resolves its sibling adaptation_manifest.json from the adapted run.
    # Path.resolve() would dereference the weights link into the base snapshot.
    path = Path(os.path.abspath(Path(path).expanduser()))
    if path.is_dir():
        transformer_dir = path
        weights_path = transformer_dir / "model.safetensors"
    else:
        weights_path = path
        transformer_dir = path.parent
    if weights_path.name != "model.safetensors" or not weights_path.is_file():
        raise FileNotFoundError(
            f"Adapted transformer must contain model.safetensors: {weights_path}"
        )
    config_path = transformer_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    manifest_path = transformer_dir.parent / "adaptation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r") as handle:
        manifest = json.load(handle)
    return transformer_dir, weights_path, manifest_path, manifest


class IVideoGPTAdaptedAdapter(IVideoGPTFrozenAdapter):
    """Use the official frozen tokenizer with a domain-adapted causal LM."""

    def __init__(
        self,
        base_checkpoint_path,
        adapted_transformer_checkpoint,
        expected_adaptation_mode,
        expected_adaptation_epoch=19,
        expected_adaptation_domain=None,
        **kwargs,
    ):
        transformer_dir, weights_path, manifest_path, manifest = (
            resolve_adapted_transformer(adapted_transformer_checkpoint)
        )
        if manifest.get("adaptation_mode") != expected_adaptation_mode:
            raise RuntimeError(
                "Adaptation mode mismatch: "
                f"{manifest.get('adaptation_mode')} != {expected_adaptation_mode}"
            )
        if manifest.get("model_family") not in (None, "ivideogpt_oxe_64_medium"):
            raise RuntimeError(
                f"Unexpected adapted model family: {manifest.get('model_family')}"
            )
        if bool(manifest.get("stopped_early", True)):
            raise RuntimeError(
                f"Refusing smoke/incomplete adapted checkpoint: {manifest_path}"
            )
        if int(manifest.get("epoch", -1)) != int(expected_adaptation_epoch):
            raise RuntimeError(
                f"Adaptation epoch mismatch: {manifest.get('epoch')} != "
                f"{expected_adaptation_epoch}"
            )
        manifest_domain = manifest.get("protocol", {}).get("domain")
        if (
            expected_adaptation_domain is not None
            and manifest_domain != expected_adaptation_domain
        ):
            raise RuntimeError(
                "Adaptation domain mismatch: "
                f"{manifest_domain} != {expected_adaptation_domain}"
            )
        if bool(manifest.get("tokenizer_trained", True)):
            raise RuntimeError("Registered adaptation requires a frozen tokenizer")
        if not bool(manifest.get("transformer_trained", False)):
            raise RuntimeError("Adaptation manifest does not mark transformer as trained")

        super().__init__(checkpoint_path=base_checkpoint_path, **kwargs)
        declared_base = Path(manifest.get("base_snapshot", "")).expanduser().resolve()
        if declared_base != self.snapshot_root:
            raise RuntimeError(
                f"Adapted checkpoint base snapshot mismatch: {declared_base} != "
                f"{self.snapshot_root}"
            )

        # The parent constructor validates and retains the official tokenizer.
        # Replace only the causal transformer; the original temporary module is
        # released before loading adapted weights to avoid a double-model peak.
        del self.transformer
        gc.collect()

        from transformers import AutoModelForCausalLM
        import transformers

        kwargs = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "output_loading_info": True,
            "attn_implementation": self.attention_implementation,
        }
        major_version = int(transformers.__version__.split(".", 1)[0])
        kwargs["dtype" if major_version >= 5 else "torch_dtype"] = self.compute_dtype
        loaded = AutoModelForCausalLM.from_pretrained(str(transformer_dir), **kwargs)
        if isinstance(loaded, tuple):
            transformer, loading_info = loaded
        else:
            transformer, loading_info = loaded, {}
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if loading_info.get(key):
                raise RuntimeError(
                    f"Adapted transformer load reported {key}: {loading_info[key]}"
                )
        if not hasattr(transformer, "model"):
            raise RuntimeError("Adapted transformer has no causal backbone at .model")
        actual_architecture = {
            "hidden_size": int(transformer.config.hidden_size),
            "num_hidden_layers": int(transformer.config.num_hidden_layers),
            "num_attention_heads": int(transformer.config.num_attention_heads),
            "vocab_size": int(transformer.config.vocab_size),
            "max_position_embeddings": int(transformer.config.max_position_embeddings),
        }
        expected_architecture = self.load_report["transformer_config"]
        if actual_architecture != expected_architecture:
            raise RuntimeError(
                "Adapted transformer architecture changed from the registered base: "
                f"{actual_architecture} != {expected_architecture}"
            )
        transformer.config.use_cache = False
        self.transformer = transformer.eval()

        self.load_report["source"] = "adapted_transformer_plus_official_tokenizer"
        self.load_report["adapted_transformer"] = {
            "weights": str(weights_path),
            "config": str(transformer_dir / "config.json"),
            "manifest": str(manifest_path),
            "adaptation_mode": expected_adaptation_mode,
            "adaptation_epoch": int(expected_adaptation_epoch),
            "adaptation_domain": manifest_domain,
            "completed_updates": manifest.get("completed_updates"),
            "protocol": _json_compatible(manifest.get("protocol")),
            "epoch_metrics": _json_compatible(manifest.get("epoch_metrics")),
            "temporary_physical_head_reused_by_probe": False,
            "loading_info": {
                key: _json_compatible(value)
                for key, value in loading_info.items()
                if key
                in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            },
        }
        self.load_report["parameter_count"]["transformer_with_lm_head"] = sum(
            parameter.numel() for parameter in self.transformer.parameters()
        )
