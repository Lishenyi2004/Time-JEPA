from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grasp_eval_rjob.config import evaluation_signature, load_config, protocol_digest
from grasp_eval_rjob.constants import FAMILY_TO_PROPERTIES, FAMILIES, PROPERTIES


def _load_pairs(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["output_dir"]).expanduser().resolve() / "scores" / "per_pair.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run score.py first; missing {path}")
    frame = pd.read_csv(path)
    frame = frame[frame["aggregation"] == "average"].copy()
    if frame["pair_id"].duplicated().any():
        raise RuntimeError(f"Duplicate average-surprise pair rows in {path}")
    return frame


def _property_mask(frame: pd.DataFrame, prop: str) -> pd.Series:
    return frame["family"].map(
        lambda family: prop in FAMILY_TO_PROPERTIES[str(family)]
    )


def _accuracy_rows(merged: pd.DataFrame, model_name: str) -> list[dict[str, Any]]:
    correct = f"correct::{model_name}"
    rows: list[dict[str, Any]] = [
        {
            "model": model_name,
            "group_type": "overall_micro",
            "group": "All",
            "n_pairs": len(merged),
            "accuracy": float(merged[correct].mean() * 100.0),
        }
    ]
    family_accuracies = []
    for family in FAMILIES:
        subset = merged[merged["family"] == family]
        accuracy = float(subset[correct].mean() * 100.0)
        family_accuracies.append(accuracy)
        rows.append(
            {
                "model": model_name,
                "group_type": "family",
                "group": family,
                "n_pairs": len(subset),
                "accuracy": accuracy,
            }
        )
    rows.append(
        {
            "model": model_name,
            "group_type": "family_macro",
            "group": "All",
            "n_pairs": len(merged),
            "accuracy": float(np.mean(family_accuracies)),
        }
    )

    property_accuracies = []
    for prop in PROPERTIES:
        subset = merged[_property_mask(merged, prop)]
        accuracy = float(subset[correct].mean() * 100.0)
        property_accuracies.append(accuracy)
        rows.append(
            {
                "model": model_name,
                "group_type": "property",
                "group": prop,
                "n_pairs": len(subset),
                "accuracy": accuracy,
            }
        )
    rows.append(
        {
            "model": model_name,
            "group_type": "property_macro",
            "group": "All",
            "n_pairs": len(merged),
            "accuracy": float(np.mean(property_accuracies)),
        }
    )
    return rows


def _stratified_delta_bootstrap(
    merged: pd.DataFrame,
    *,
    time_column: str,
    video_column: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired pair bootstrap, stratified within each of the 16 families."""
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    family_means: dict[str, np.ndarray] = {}
    family_sizes: dict[str, int] = {}
    for family in FAMILIES:
        frame = merged[merged["family"] == family]
        delta = (
            frame[time_column].to_numpy(dtype=np.float64)
            - frame[video_column].to_numpy(dtype=np.float64)
        )
        if not len(delta):
            raise RuntimeError(f"No pairs for family {family}")
        draws = rng.integers(0, len(delta), size=(samples, len(delta)))
        family_means[family] = delta[draws].mean(axis=1)
        family_sizes[family] = len(delta)

    property_distributions: dict[str, np.ndarray] = {}
    for prop in PROPERTIES:
        families = [
            family for family in FAMILIES if prop in FAMILY_TO_PROPERTIES[family]
        ]
        denominator = sum(family_sizes[family] for family in families)
        property_distributions[prop] = sum(
            family_means[family] * family_sizes[family] for family in families
        ) / denominator
    macro = np.stack(
        [property_distributions[prop] for prop in PROPERTIES], axis=0
    ).mean(axis=0)
    micro = sum(
        family_means[family] * family_sizes[family] for family in FAMILIES
    ) / sum(family_sizes.values())
    return {
        "property_macro_ci95_pp": [
            float(value * 100.0) for value in np.quantile(macro, [0.025, 0.975])
        ],
        "overall_micro_ci95_pp": [
            float(value * 100.0) for value in np.quantile(micro, [0.025, 0.975])
        ],
        "property_ci95_pp": {
            prop: [
                float(value * 100.0)
                for value in np.quantile(property_distributions[prop], [0.025, 0.975])
            ]
            for prop in PROPERTIES
        },
    }


def compare(
    official_config: dict[str, Any],
    video_config: dict[str, Any],
    time_config: dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    configs = [official_config, video_config, time_config]
    if len({evaluation_signature(config) for config in configs}) != 1:
        raise RuntimeError("Model evaluation protocols or frozen datasets differ")
    names = [str(config["name"]) for config in configs]
    if len(set(names)) != 3:
        raise ValueError(f"Model names must be unique: {names}")

    frames = []
    metadata_columns = ["pair_id", "family", "properties"]
    for config in configs:
        name = str(config["name"])
        frame = _load_pairs(config)
        frame = frame[
            metadata_columns
            + [
                "possible_surprise",
                "impossible_surprise",
                "margin",
                "correct",
            ]
        ].rename(
            columns={
                "possible_surprise": f"possible::{name}",
                "impossible_surprise": f"impossible::{name}",
                "margin": f"margin::{name}",
                "correct": f"correct::{name}",
            }
        )
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on=metadata_columns,
            how="inner",
            validate="one_to_one",
        )
    if any(len(frame) != len(merged) for frame in frames):
        raise RuntimeError(
            "Cross-model pair intersection is incomplete: "
            f"merged={len(merged)}, inputs={[len(frame) for frame in frames]}"
        )
    if sorted(merged["family"].unique()) != sorted(FAMILIES):
        raise RuntimeError("Merged results do not contain all 16 GRASP families")

    rows = []
    for name in names:
        rows.extend(_accuracy_rows(merged, name))
    metrics = pd.DataFrame(rows)
    video_name = str(video_config["name"])
    time_name = str(time_config["name"])
    reference_accuracy = metrics.loc[
        metrics["model"] == video_name,
        ["group_type", "group", "accuracy"],
    ].rename(columns={"accuracy": "video_accuracy"})
    metrics = metrics.merge(
        reference_accuracy, on=["group_type", "group"], validate="many_to_one"
    )
    metrics["delta_vs_video_pp"] = metrics["accuracy"] - metrics["video_accuracy"]
    metrics = metrics.drop(columns=["video_accuracy"])

    time_column = f"correct::{time_name}"
    video_column = f"correct::{video_name}"
    bootstrap = _stratified_delta_bootstrap(
        merged,
        time_column=time_column,
        video_column=video_column,
        samples=bootstrap_samples,
        seed=seed,
    )
    property_deltas = {
        prop: float(
            (
                merged.loc[_property_mask(merged, prop), time_column].mean()
                - merged.loc[_property_mask(merged, prop), video_column].mean()
            )
            * 100.0
        )
        for prop in PROPERTIES
    }
    primary_rows = metrics[
        (metrics["group_type"] == "property_macro")
        & metrics["model"].isin([video_name, time_name])
    ].set_index("model")
    primary_video = float(primary_rows.loc[video_name, "accuracy"])
    primary_time = float(primary_rows.loc[time_name, "accuracy"])
    scoring = official_config["scoring"]
    largest_gain_property = max(property_deltas, key=property_deltas.get)
    leave_largest_gain_out_delta = float(
        np.mean(
            [
                delta
                for prop, delta in property_deltas.items()
                if prop != largest_gain_property
            ]
        )
    )
    decision: dict[str, Any] = {
        "protocol_digests": {
            str(config["name"]): protocol_digest(config) for config in configs
        },
        "evaluation_signature": evaluation_signature(official_config),
        "n_pairs": int(len(merged)),
        "primary_metric": "8-property equal macro pair accuracy (AverageSurprise)",
        "primary_video_accuracy": primary_video,
        "primary_time_accuracy": primary_time,
        "primary_delta_time_minus_video_pp": primary_time - primary_video,
        "primary_delta_bootstrap_ci95_pp": bootstrap["property_macro_ci95_pp"],
        "overall_micro_delta_bootstrap_ci95_pp": bootstrap["overall_micro_ci95_pp"],
        "property_deltas_time_minus_video_pp": property_deltas,
        "property_delta_bootstrap_ci95_pp": bootstrap["property_ci95_pp"],
        "nondegraded_properties_delta_ge_0": int(
            sum(delta >= 0 for delta in property_deltas.values())
        ),
        "strictly_positive_properties_delta_gt_0": int(
            sum(delta > 0 for delta in property_deltas.values())
        ),
        "largest_gain_property": largest_gain_property,
        "leave_largest_gain_out_macro_delta_pp": leave_largest_gain_out_delta,
        "thresholds": {
            "macro_delta_gt_0": True,
            "minimum_nondegraded_properties": int(
                scoring["minimum_nondegraded_properties"]
            ),
            "require_leave_largest_gain_out_positive": bool(
                scoring["require_leave_largest_gain_out_positive"]
            ),
        },
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "unit": "paired possible/impossible pair",
            "stratification": "within each of 16 families",
            "paired_across_models": True,
            "excludes_encoder_training_seed_uncertainty": True,
        },
        "aggregation_correction": (
            "Eight non-empty properties follow the released JEPA mapping and the "
            "original GRASP paper. The later JEPA paper's count of ten is inconsistent "
            "with its public aggregation code. The draft 7/10 non-degradation rule is "
            "translated before model results to ceil(70% * 8) = 6/8. The macro "
            "must remain positive after excluding the largest-gain property."
        ),
    }
    decision["go"] = bool(
        decision["primary_delta_time_minus_video_pp"] > 0
        and decision["nondegraded_properties_delta_ge_0"]
        >= decision["thresholds"]["minimum_nondegraded_properties"]
        and decision["leave_largest_gain_out_macro_delta_pp"] > 0
    )
    return metrics, merged, decision


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-p{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-config", required=True)
    parser.add_argument("--video-config", required=True)
    parser.add_argument("--time-config", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    official = load_config(args.official_config)
    video = load_config(args.video_config)
    time = load_config(args.time_config)
    metrics, merged, decision = compare(
        official,
        video,
        time,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output_dir).expanduser().resolve()
    _atomic_text(output / "comparison.csv", metrics.to_csv(index=False))
    _atomic_text(output / "paired_predictions.csv", merged.to_csv(index=False))
    _atomic_text(
        output / "decision.json",
        json.dumps(decision, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    print(metrics.to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
