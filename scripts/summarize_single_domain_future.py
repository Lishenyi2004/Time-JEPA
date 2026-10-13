#!/usr/bin/env python3
"""Audit single-domain multi-horizon future-state probes against mix_e19."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "covla_probe_rjob/runs/single_domain_future_20260915"
SINGLE_ROOT = ROOT / "time-jepa/outputs_future_state/single_domain_1s_5step"
MIX_ROOT = ROOT / "time-jepa/outputs_future_state/covla_agibot_egoexo_1s_5step/mix_e19"
HORIZONS = ("0.2", "0.4", "0.6", "0.8", "1")
METRICS = ("nrmse", "r2", "pearson")
DOMAINS = {
    "covla": {"single": "covla_e19", "label": "CoVLA"},
    "agibot": {"single": "agibot_e17", "label": "AgiBot"},
    "egoexo": {"single": "egoexo_e19", "label": "EgoExo4D"},
}
GROUPS = {
    "Seen": {"single": "seen", "mix": "readout"},
    "Unseen": {"single": "unseen", "mix": "shot"},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_result(path: Path) -> Path:
    matches = list(path.glob("*/test_metrics.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one test_metrics.json below {path}, got {matches}")
    return matches[0]


def result_paths(kind: str, domain: str, group: str) -> Path:
    if kind == "single_domain":
        return one_result(SINGLE_ROOT / DOMAINS[domain]["single"] / GROUPS[group]["single"])
    return one_result(MIX_ROOT / domain / GROUPS[group]["mix"])


def metric_blocks(report: dict, domain: str, split: str) -> dict:
    if split == "ID":
        return {"test": report["test"]}
    if domain == "covla":
        return {"weather_ood_test": report["ood_test"]}
    return report["ood_view_test"]


def selected_indices(report: dict, domain: str, group: str, split: str, views) -> dict:
    count = len(report["target_fields"])
    selected = {view: list(range(count)) for view in views}
    if (domain, group, split) == ("agibot", "Unseen", "OOD"):
        selected = {
            view: ([0, 2, 4, 6] if "left" in view else [1, 3, 5, 7])
            for view in views
        }
    return selected


def aggregate(report: dict, domain: str, group: str, split: str) -> dict:
    blocks = metric_blocks(report, domain, split)
    selected = selected_indices(report, domain, group, split, blocks)
    per_horizon = {}
    for horizon in HORIZONS:
        per_horizon[horizon] = {}
        for metric in METRICS:
            per_view = []
            for view, block in blocks.items():
                indices = selected[view]
                values = [block["per_horizon"][horizon][metric][index] for index in indices]
                counts = [block["per_horizon"][horizon]["valid_count"][index] for index in indices]
                if not values or not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite {metric}: {domain}/{group}/{split}/{view}/{horizon}")
                if not all(count > 0 for count in counts):
                    raise ValueError(f"No valid labels: {domain}/{group}/{split}/{view}/{horizon}")
                per_view.append(mean(values))
            per_horizon[horizon][metric] = mean(per_view)
    return {
        "protocol": "same_side_visibility" if (domain, group, split) == ("agibot", "Unseen", "OOD") else "full_target",
        "target_indices_by_view": selected,
        "per_horizon": per_horizon,
        "horizon_macro": {
            metric: mean(per_horizon[horizon][metric] for horizon in HORIZONS)
            for metric in METRICS
        },
    }


def assert_matched(single: dict, mixed: dict, domain: str, group: str) -> None:
    if single["target_fields"] != mixed["target_fields"]:
        raise ValueError(f"Target mismatch: {domain}/{group}")
    for split in ("ID", "OOD"):
        single_blocks = metric_blocks(single, domain, split)
        mixed_blocks = metric_blocks(mixed, domain, split)
        if single_blocks.keys() != mixed_blocks.keys():
            raise ValueError(f"View mismatch: {domain}/{group}/{split}")
        for view in single_blocks:
            for horizon in HORIZONS:
                single_block = single_blocks[view]["per_horizon"][horizon]
                mixed_block = mixed_blocks[view]["per_horizon"][horizon]
                if single_block["valid_count"] != mixed_block["valid_count"]:
                    raise ValueError(f"Valid-count mismatch: {domain}/{group}/{split}/{view}/{horizon}")
                for metric in ("mae", "rmse", "nrmse", "valid_count"):
                    left = single_block["baselines"]["train_mean"][metric]
                    right = mixed_block["baselines"]["train_mean"][metric]
                    if len(left) != len(right) or any(abs(a - b) > 1e-9 for a, b in zip(left, right)):
                        raise ValueError(
                            f"Train-mean baseline mismatch: {domain}/{group}/{split}/{view}/{horizon}/{metric}"
                        )


def main() -> None:
    records = []
    sources = []
    loaded = {}
    for domain in DOMAINS:
        for group in GROUPS:
            for kind in ("single_domain", "mix_e19"):
                path = result_paths(kind, domain, group)
                report = json.loads(path.read_text())
                if report["target_mode"] != "future_state":
                    raise ValueError(f"Wrong target mode: {path}")
                if report["future_horizons_seconds"] != [0.2, 0.4, 0.6, 0.8, 1.0]:
                    raise ValueError(f"Wrong horizons: {path}")
                loaded[kind, domain, group] = report
                sources.append({
                    "encoder": kind,
                    "domain": domain,
                    "target_group": group,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest(path),
                })
            assert_matched(
                loaded["single_domain", domain, group],
                loaded["mix_e19", domain, group],
                domain,
                group,
            )
            for split in ("ID", "OOD"):
                row = {
                    "domain": domain,
                    "target_group": group,
                    "split": split,
                    "target_count": len(loaded["single_domain", domain, group]["target_fields"]),
                }
                for kind in ("single_domain", "mix_e19"):
                    summary = aggregate(loaded[kind, domain, group], domain, group, split)
                    row[kind] = summary
                row["delta_mix_minus_single"] = {
                    metric: row["mix_e19"]["horizon_macro"][metric]
                    - row["single_domain"]["horizon_macro"][metric]
                    for metric in METRICS
                }
                records.append(row)

    aggregates = []
    for group in GROUPS:
        for split in ("ID", "OOD"):
            rows = [row for row in records if row["target_group"] == group and row["split"] == split]
            count = sum(row["target_count"] for row in rows)
            item = {"target_group": group, "split": split, "target_count": count}
            for kind in ("single_domain", "mix_e19"):
                item[kind] = {
                    metric: sum(
                        row["target_count"] * row[kind]["horizon_macro"][metric]
                        for row in rows
                    ) / count
                    for metric in METRICS
                }
            item["delta_mix_minus_single"] = {
                metric: item["mix_e19"][metric] - item["single_domain"][metric]
                for metric in METRICS
            }
            aggregates.append(item)

    payload = {
        "audit_date": "2026-09-16",
        "horizons_seconds": [0.2, 0.4, 0.6, 0.8, 1.0],
        "aggregation": (
            "Equal mean over targets within each view, views within a split, and five horizons. "
            "AgiBot Unseen OOD uses left-fisheye indices [0,2,4,6] and right-fisheye "
            "indices [1,3,5,7]; AgiBot Seen OOD retains all 18 targets."
        ),
        "comparison_check": (
            "Single-domain and mix_e19 reports match in target fields, evaluation views, "
            "per-horizon valid counts, and train-mean baseline arrays."
        ),
        "records": records,
        "dimension_weighted_aggregates": aggregates,
        "sources": sources,
    }
    result_path = RUN / "results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    with (RUN / "results.csv").open("w", newline="") as handle:
        fields = [
            "domain", "target_group", "split", "target_count", "protocol", "encoder",
            *METRICS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            for kind in ("single_domain", "mix_e19"):
                writer.writerow({
                    "domain": row["domain"],
                    "target_group": row["target_group"],
                    "split": row["split"],
                    "target_count": row["target_count"],
                    "protocol": row[kind]["protocol"],
                    "encoder": kind,
                    **row[kind]["horizon_macro"],
                })
    print(f"Wrote {result_path} and {RUN / 'results.csv'}")


if __name__ == "__main__":
    main()
