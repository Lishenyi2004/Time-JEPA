#!/usr/bin/env python3
"""Export future-state ID/OOD tables from the 18 completed five-horizon runs.

The 2026-09-10 reporting extension applies the existing AgiBot same-side
visibility rule to future HO-OOD at every horizon. Original metrics are read
only; full-target and cross-side summaries remain in the audit JSON.
"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "time-jepa/outputs_future_state/covla_agibot_egoexo_1s_5step"
OUT = ROOT / "time-jepa/outputs_future_state/reports/visibility_matched_20260910"
sys.path.insert(0, str(ROOT / "covla_probe_rjob"))
from probe_view_metrics import (  # noqa: E402
    AGIBOT_FISHEYE_VIEW_SIDES,
    validate_agibot_same_side_ood,
)

HORIZONS = ("0.2", "0.4", "0.6", "0.8", "1")
METRICS = ("nrmse", "r2", "pearson")
ENCODERS = {"zero": "Original", "video_only_e19": "Video FT", "mix_e19": "Time-JEPA"}
DOMAINS = {"covla": "autonomous driving", "agibot": "robot manipulation", "egoexo": "human motion"}
TARGET_GROUPS = {"readout": "FT", "shot": "HO"}


def aggregate(blocks, indices):
    """Average targets within each view, then views, then the five horizons."""
    per_horizon = {}
    for h in HORIZONS:
        per_horizon[h] = {}
        for metric in METRICS:
            view_means = []
            for view, block in blocks.items():
                data = block["per_horizon"][h]
                values = [data[metric][i] for i in indices[view]]
                if not values or not all(math.isfinite(x) for x in values):
                    raise ValueError(f"Invalid metric: {view}/{h}/{metric}")
                if any(data["valid_count"][i] <= 0 for i in indices[view]):
                    raise ValueError(f"No valid labels: {view}/{h}")
                view_means.append(mean(values))
            per_horizon[h][metric] = mean(view_means)
    return {
        "per_horizon": per_horizon,
        "horizon_macro": {m: mean(per_horizon[h][m] for h in HORIZONS) for m in METRICS},
    }


def collect():
    records, diagnostics, sources, schemas = [], {}, [], {}
    for encoder in ENCODERS:
        for domain in DOMAINS:
            for task, group in TARGET_GROUPS.items():
                paths = list((SOURCE / encoder / domain / task).glob("*/test_metrics.json"))
                if len(paths) != 1:
                    raise ValueError(f"Expected one result for {encoder}/{domain}/{task}")
                path = paths[0]
                raw = path.read_bytes()
                report = json.loads(raw)
                assert report["target_mode"] == "future_state"
                assert report["future_horizons_seconds"] == [float(h) for h in HORIZONS]
                fields = report["target_fields"]
                schema = (domain, task)
                if schema in schemas:
                    assert fields == schemas[schema], f"Target mismatch: {schema}"
                schemas[schema] = fields
                sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()})
                for split in ("ID", "OOD"):
                    if split == "ID":
                        blocks = {"test": report["test"]}
                    elif domain == "covla":
                        blocks = {"weather_ood_test": report["ood_test"]}
                    else:
                        blocks = report["ood_view_test"]
                    assert blocks
                    indices = {view: list(range(len(fields))) for view in blocks}
                    full = aggregate(blocks, indices)
                    # Check recomputation against the cached target/horizon macros.
                    for metric in METRICS:
                        cached = mean(mean(b["horizon_macro"][metric]) for b in blocks.values())
                        assert math.isclose(full["horizon_macro"][metric], cached, abs_tol=1e-6)
                    summary, protocol = full, "full_target"
                    if (domain, task, split) == ("agibot", "shot", "OOD"):
                        sides = validate_agibot_same_side_ood(fields, blocks.keys())
                        assert sides == {"left": [0, 2, 4, 6], "right": [1, 3, 5, 7]}
                        indices = {v: sides[s] for v, s in AGIBOT_FISHEYE_VIEW_SIDES.items()}
                        cross_indices = {v: sides["right" if s == "left" else "left"] for v, s in AGIBOT_FISHEYE_VIEW_SIDES.items()}
                        summary, protocol = aggregate(blocks, indices), "same_side_visibility"
                        cross = aggregate(blocks, cross_indices)
                        for h in HORIZONS:
                            for metric in METRICS:
                                reconstructed = (summary["per_horizon"][h][metric] + cross["per_horizon"][h][metric]) / 2
                                assert math.isclose(reconstructed, full["per_horizon"][h][metric], abs_tol=1e-12)
                        diagnostics[encoder] = {"full_8d": full, "cross_side": cross}
                    records.append({
                        "encoder": encoder, "dataset": domain, "domain": DOMAINS[domain],
                        "target_group": group, "split": split, "protocol": protocol,
                        "target_count": len(fields), "source": str(path.relative_to(ROOT)),
                        "target_indices_by_view": indices,
                        "selected_targets_by_view": {v: [fields[i] for i in idx] for v, idx in indices.items()},
                        **summary,
                    })
    return records, diagnostics, sources


def get_record(records, domain, group, split, encoder):
    return next(r for r in records if (r["dataset"], r["target_group"], r["split"], r["encoder"]) == (domain, group, split, encoder))


def markdown_table(records, split):
    lines = ["| 数据集 | 目标 | Original | Video FT | Time-JEPA |", "|---|---|---:|---:|---:|"]
    names = {"covla": "CoVLA", "agibot": "AgiBot", "egoexo": "EgoExo4D"}
    for domain in DOMAINS:
        for group in TARGET_GROUPS.values():
            metrics = [get_record(records, domain, group, split, e)["horizon_macro"] for e in ENCODERS]
            best = {"nrmse": min(m["nrmse"] for m in metrics), "r2": max(m["r2"] for m in metrics)}
            cells = []
            for values in metrics:
                cells.append(" / ".join(f"**{values[m]:.3f}**" if values[m] == best[m] else f"{values[m]:.3f}" for m in ("nrmse", "r2")))
            lines.append("| " + " | ".join([names[domain], group, *cells]) + " |")
    return "\n".join(lines)


def draw_tables(records, splits, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    height = 8.2 if len(splits) == 2 else 4.9
    fig = plt.figure(figsize=(12.8, height), facecolor="white")
    fig.text(.03, .955, "Future-state prediction: " + " and ".join(splits), fontsize=18, weight="bold")
    fig.text(.03, .915 if len(splits) == 2 else .89, "Equal mean over five horizons (0.2, 0.4, 0.6, 0.8, 1.0 s)", fontsize=11, color="#444444")
    for panel, split in enumerate(splits):
        y, panel_height = ((.54 if panel == 0 else .195), .31) if len(splits) == 2 else (.285, .50)
        ax = fig.add_axes([.03, y, .94, panel_height])
        ax.axis("off")
        ax.text(0, 1.04, split, transform=ax.transAxes, fontsize=13, weight="bold", color="#174B63")
        headers = ["Domain", "Targets"] + [f"{label}\n{metric}" for label in ENCODERS.values() for metric in ("nRMSE ↓", "R² ↑")]
        rows, best_cells = [], []
        for domain in DOMAINS:
            for group in TARGET_GROUPS.values():
                values = [get_record(records, domain, group, split, e)["horizon_macro"] for e in ENCODERS]
                rows.append([DOMAINS[domain], group] + [f"{v[m]:.3f}" for v in values for m in ("nrmse", "r2")])
                for metric_index, metric in enumerate(("nrmse", "r2")):
                    best = (min if metric == "nrmse" else max)(v[metric] for v in values)
                    best_cells.extend((len(rows), 2 + 2*i + metric_index) for i, v in enumerate(values) if v[metric] == best)
        table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", colWidths=[.23, .075] + [.695/6]*6, bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        for (row, col), cell in table.get_celld().items():
            cell.visible_edges = "horizontal"
            cell.set_edgecolor("#B7C1C6")
            cell.set_linewidth(.5 if row in (0, 2, 4, 6) else .2)
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_height(cell.get_height()*1.35)
            if col == 0:
                cell.set_text_props(ha="left")
                cell.PAD = .04
        for cell in best_cells:
            table[cell].set_text_props(weight="bold")
    notes = [
        "FT = fine-tuning targets; HO = held-out targets. Bold marks the best configuration for each metric.",
        "ID: CoVLA test, AgiBot top-head view, EgoExo4D view01. OOD: rain, two fisheye views, held-out view03.",
        "AgiBot HO-OOD: left fisheye → four left-arm targets; right fisheye → four right-arm targets.",
        "AgiBot FT-OOD uses all 18 targets in each fisheye view. Targets and views receive equal weight.",
    ]
    for i, note in enumerate(notes):
        fig.text(.03, .145 - i*.032 if len(splits) == 2 else .205 - i*.047, note, fontsize=9.2, color="#333333")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=220, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def main():
    records, diagnostics, sources = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "reporting_revision_date": "2026-09-10",
        "selection": "Author-requested extension of existing current-state HO-OOD visibility rule after inspection of full-target future OOD results.",
        "id_scope": "Existing paper protocol: CoVLA test, AgiBot top_head, EgoExo4D view01.",
        "horizons_seconds": [float(h) for h in HORIZONS],
        "records": records, "agibot_ho_ood_diagnostics": diagnostics, "sources": sources,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    csv_fields = ["dataset", "domain", "target_group", "split", "encoder", "protocol", *METRICS]
    for filename, per_horizon in (("summary.csv", False), ("per_horizon.csv", True)):
        with (OUT / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields + (["horizon_seconds"] if per_horizon else []))
            writer.writeheader()
            for record in records:
                base = {k: record[k] for k in csv_fields if k not in METRICS}
                if per_horizon:
                    for h, metrics in record["per_horizon"].items():
                        writer.writerow({**base, **metrics, "horizon_seconds": float(h)})
                else:
                    writer.writerow({**base, **record["horizon_macro"]})
    readme = "# Future-state ID/OOD：五时距等权汇总\n\n每格为 **nRMSE ↓ / R² ↑**；逐指标用未舍入值确定最优项。FT/HO 分别表示进入过/未进入 encoder 时间监督的目标。\n\n"
    readme += "ID 沿用论文的 CoVLA test、AgiBot top_head、EgoExo4D view01。OOD 为 CoVLA 雨天、AgiBot 两个鱼眼视角、EgoExo4D view03。AgiBot HO-OOD 在每个 horizon 内按左鱼眼配左臂四项、右鱼眼配右臂四项等权汇总，再对五个 horizon 等权；FT-OOD 仍为两视角完整 18D。\n\n"
    for split in ("ID", "OOD"):
        readme += f"## {split}\n\n" + markdown_table(records, split) + "\n\n"
    readme += "2026-09-10 按作者要求，在已查看 full-target future OOD 结果后采用上述可见性汇总。原始 test_metrics.json 保持完整；summary.json 保留 full-8D/cross-side 汇总、逐 horizon 指标、目标选择和 18 个源文件的 SHA-256。AgiBot HO-OOD 的 nRMSE 改善，但 R² 仍低于 Video FT。\n\n复现：`python scripts/summarize_future_state_ood.py`。\n"
    (OUT / "README.md").write_text(readme)
    draw_tables(records, ("ID", "OOD"), "future_state_id_ood")
    draw_tables(records, ("OOD",), "future_state_ood")
    print(f"Validated {len(sources)} sources; exported {len(records)} ID/OOD rows to {OUT}")
    print(markdown_table(records, "OOD"))


if __name__ == "__main__":
    main()
