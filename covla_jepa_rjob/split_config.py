"""Pure helpers for resolving encoder-train splits from YAML configs.

This module intentionally has no torch/application imports so the submission
wrapper and unit tests use exactly the same split-resolution rules.
"""

from pathlib import Path


_DATASET_TYPE_TO_DOMAIN = {
    "covlavideodataset": "covla",
    "covla_video": "covla",
    "covla": "covla",
    "covlatimejepadataset": "covla",
    "covla_time_jepa": "covla",
    "covla_time": "covla",
    "agibottimejepadataset": "agibot",
    "agibot_time_jepa": "agibot",
    "agibot_time": "agibot",
    "egoexotimejepadataset": "egoexo",
    "egoexo_time_jepa": "egoexo",
    "egoexo_time": "egoexo",
}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _unique_preserve_order(values):
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def top_level_dataset_domain(params):
    data_cfg = params.get("data", {}) or {}
    dataset_type = str(data_cfg.get("dataset_type", "")).strip().lower()
    return _DATASET_TYPE_TO_DOMAIN.get(dataset_type)


def configured_encoder_domains(params):
    domains = set()
    top_level_domain = top_level_dataset_domain(params)
    if top_level_domain:
        domains.add(top_level_domain)

    time_jepa_cfg = params.get("time_jepa", {}) or {}
    for config_name, domain_cfg in (time_jepa_cfg.get("domains", {}) or {}).items():
        if not isinstance(domain_cfg, dict):
            continue
        domains.add(str(domain_cfg.get("domain", config_name)).strip().lower())
    return domains


def _domain_configs(params, domain_name):
    target_domain = str(domain_name).strip().lower()
    configs = []

    data_cfg = params.get("data", {}) or {}
    if top_level_dataset_domain(params) == target_domain:
        configs.append(data_cfg)

    time_jepa_cfg = params.get("time_jepa", {}) or {}
    for config_name, domain_cfg in (time_jepa_cfg.get("domains", {}) or {}).items():
        if not isinstance(domain_cfg, dict):
            continue
        configured_domain = str(
            domain_cfg.get("domain", config_name)
        ).strip().lower()
        if configured_domain == target_domain:
            configs.append(domain_cfg)
    return configs


def configured_encoder_train_split(params, domain_name, repo_root=None):
    split_paths = [
        str(domain_cfg["encoder_train_split"])
        for domain_cfg in _domain_configs(params, domain_name)
        if domain_cfg.get("encoder_train_split")
    ]
    split_paths = _unique_preserve_order(split_paths)
    if len(split_paths) > 1:
        raise RuntimeError(
            f"Conflicting encoder_train_split values for domain={domain_name}: "
            f"{split_paths}"
        )
    if not split_paths:
        return None

    split_path = Path(split_paths[0]).expanduser()
    if not split_path.is_absolute() and repo_root is not None:
        split_path = Path(repo_root) / split_path
    return str(split_path.resolve())


def encoder_train_split_required(params, domain_name):
    values = {
        bool(domain_cfg.get("require_encoder_train_split", False))
        for domain_cfg in _domain_configs(params, domain_name)
    }
    return True in values


def expected_encoder_train_videos(params, domain_name):
    values = [
        int(domain_cfg["expected_encoder_train_videos"])
        for domain_cfg in _domain_configs(params, domain_name)
        if domain_cfg.get("expected_encoder_train_videos") is not None
    ]
    values = _unique_preserve_order(values)
    if len(values) > 1:
        raise RuntimeError(
            f"Conflicting expected_encoder_train_videos values for "
            f"domain={domain_name}: {values}"
        )
    if not values:
        return None
    if values[0] <= 0:
        raise RuntimeError(
            "expected_encoder_train_videos must be positive, "
            f"got {values[0]} for domain={domain_name}"
        )
    return values[0]


def resolve_encoder_split_enabled(flag, params, domain_name):
    normalized_flag = str(flag if flag is not None else "auto").strip().lower()
    if normalized_flag in _TRUE_VALUES:
        enabled = True
    elif normalized_flag in _FALSE_VALUES:
        enabled = False
    elif normalized_flag == "auto":
        enabled = str(domain_name).strip().lower() in configured_encoder_domains(params)
    else:
        raise ValueError(
            f"Invalid ENABLE_{str(domain_name).upper()}_SPLIT value: {flag!r}"
        )

    if encoder_train_split_required(params, domain_name) and not enabled:
        raise RuntimeError(
            f"{domain_name} config requires an encoder-train split, but split "
            f"injection was disabled by flag={flag!r}"
        )
    return enabled
