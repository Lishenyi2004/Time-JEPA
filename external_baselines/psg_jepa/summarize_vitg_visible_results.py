"""Apply same-side arm visibility to every PSG-g AgiBot OOD panel.

Read only the completed probe artifacts. Keep ID aggregation unchanged and
retain full-target/cross-side diagnostics for this reporting revision.
"""

from __future__ import annotations

import csv
import json
from statistics import mean

from summarize_vitg_results import (
    HERE, HORIZONS, METRICS, MODELS, MODEL_LABELS, TIME_JEPA_NOTICE,
    close, load_probe, macro, read_json,
)


FISHEYE = {
    "left": "observation.images.head_left_fisheye",
    "right": "observation.images.head_right_fisheye",
}
SHARED_FIELDS = [
    "state/robot/translation_speed", "state/robot/angular_speed",
    "state/waist/position_pc1", "state/waist/position_pc2",
]


def side_indices(fields, task):
    if task == "shot":
        selected = {side: [i for i, field in enumerate(fields) if field.endswith(f"_{side}")]
                    for side in FISHEYE}
        assert selected == {"left": [0, 2, 4, 6], "right": [1, 3, 5, 7]}
    else:
        selected = {}
        for side, coordinates in (("left", range(3)), ("right", range(3, 6))):
            names = ([f"state/end/arm_position[{i}]" for i in coordinates]
                     + [f"state/end/arm_position_velocity_world[{i}]" for i in coordinates]
                     + [f"state/{side}_effector/position"])
            selected[side] = [fields.index(name) for name in names]
        assert selected == {"left": [4, 5, 6, 10, 11, 12, 16],
                            "right": [7, 8, 9, 13, 14, 15, 17]}
        assert [f for i, f in enumerate(fields) if i not in selected["left"] + selected["right"]] == SHARED_FIELDS
    return selected


def selections(data, task, split):
    fields = data["target_fields"]
    views = data["id_view_test" if split == "ID" else "ood_view_test"]
    full = {view: list(range(len(fields))) for view in views}
    if split == "ID":
        assert set(views) == {"observation.images.top_head",
                              "observation.images.hand_left", "observation.images.hand_right"}
        return {"primary": full}
    assert set(views) == set(FISHEYE.values())
    sides = side_indices(fields, task)
    result = {
        "primary": {view: sides[side] for side, view in FISHEYE.items()},
        "cross_side": {view: sides["right" if side == "left" else "left"]
                       for side, view in FISHEYE.items()},
        "full_target": full,
    }
    if task != "shot":
        result["shared_4d"] = {view: [fields.index(field) for field in SHARED_FIELDS]
                               for view in views}
    return result


def aggregate(data, split, chosen, horizon=None):
    views = data["id_view_test" if split == "ID" else "ood_view_test"]
    indices = sorted(set(index for selection in chosen.values() for index in selection))
    arrays = {metric: [] for metric in (*METRICS, "rmse")}
    for index in indices:
        blocks = [block if horizon is None else block["per_horizon"][horizon]
                  for view, block in views.items() if index in chosen[view]]
        assert blocks
        for block in blocks:
            if "valid_count" in block:
                assert block["valid_count"][index] > 0
        for metric in arrays:
            arrays[metric].append(mean(block[metric][index] for block in blocks))
    # All chosen views have equal panel size, so this agrees with view-first
    # macro averaging while retaining one auditable value per target.
    assert len(set(len(value) for value in chosen.values())) == 1
    return indices, arrays


def main():
    legacy = read_json(HERE / "vitg_results_summary.json")
    audited_inputs = {(row["model"], row["task"]): row for row in legacy["inputs"]}
    records, sources, cells, diagnostics, selections_by_panel = [], [], [], [], []
    for task in ("readout", "shot", "future"):
        data_by_model = {}
        for model in MODELS:
            data, audit = load_probe(model, task)
            assert audit["sha256"] == audited_inputs[model, task]["sha256"]
            data_by_model[model] = data
            sources.append(audit)
        reference = data_by_model["Time-JEPA"]
        for model, data in data_by_model.items():
            close(data["target_fields"], reference["target_fields"])
            close(data["splits"], reference["splits"])
        for split in ("ID", "OOD"):
            chosen = selections(reference, task, split)
            selection = chosen["primary"]
            selected_indices = sorted(set(i for values in selection.values() for i in values))
            selections_by_panel.append({
                "task": task, "split": split, "target_indices_by_view": selection,
                "selected_fields_by_view": {view: [reference["target_fields"][i] for i in values]
                                            for view, values in selection.items()},
                "excluded_fields": [field for i, field in enumerate(reference["target_fields"])
                                    if i not in selected_indices],
            })
            primary_arrays = {}
            for model, data in data_by_model.items():
                per_horizon = {}
                diagnostic_horizons = {name: {} for name in chosen if name != "primary"}
                for horizon in (HORIZONS if task == "future" else (None,)):
                    indices, arrays = aggregate(data, split, selection, horizon)
                    assert indices == selected_indices
                    per_horizon[horizon] = macro(arrays)
                    primary_arrays[model, horizon] = arrays
                    for name, choice in chosen.items():
                        if name != "primary":
                            _, diagnostic = aggregate(data, split, choice, horizon)
                            diagnostic_horizons[name][horizon] = macro(diagnostic)
                    if split == "OOD":
                        for metric in METRICS:
                            same = per_horizon[horizon][metric]
                            cross = diagnostic_horizons["cross_side"][horizon][metric]
                            full = diagnostic_horizons["full_target"][horizon][metric]
                            reconstructed = ((same + cross) / 2 if task == "shot" else
                                             (7 * same + 7 * cross + 4 * diagnostic_horizons["shared_4d"][horizon][metric]) / 18)
                            close(reconstructed, full, tolerance=1e-12)
                values = {metric: mean(row[metric] for row in per_horizon.values()) for metric in METRICS}
                previous = next(row["models"][model] for row in legacy["panels"]
                                if (row["task"], row["split"]) == (task, split))
                if split == "ID" or task == "shot":
                    close(values, previous)
                else:
                    close({metric: mean(row[metric] for row in diagnostic_horizons["full_target"].values())
                           for metric in METRICS}, previous)
                records.append({"task": task, "split": split, "model": model,
                                "target_count": len(indices), **values,
                                "per_horizon": per_horizon if task == "future" else None,
                                "change_from_previous": {metric: values[metric] - previous[metric]
                                                         for metric in METRICS}})
                if diagnostic_horizons:
                    diagnostics.append({"task": task, "split": split, "model": model,
                                        "per_horizon": diagnostic_horizons})
            if task == "future":
                for horizon in HORIZONS:
                    for position, index in enumerate(selected_indices):
                        cells.append({"split": split, "horizon": float(horizon),
                                      "target": reference["targets"][index],
                                      "target_field": reference["target_fields"][index],
                                      **{model: primary_arrays[model, horizon]["rmse"][position]
                                         for model in MODELS}})
    payload = {
        "reporting_revision_date": "2026-09-10",
        "model_labels": MODEL_LABELS,
        "time_jepa_experiment_scope": legacy["time_jepa_experiment_scope"],
        "protocol": "same_side_arm_ood_v1",
        "selection": "User-requested reporting extension: every OOD panel uses only same-side arm targets; remove non-lateralized robot/waist targets from Readout/Future; keep ID unchanged.",
        "aggregation": "equal selected target, equal view; future additionally equal five horizons",
        "audit": "9 unchanged metric SHA256 values and complete histories; verify legacy aggregates and same/cross/shared decomposition",
        "records": records, "selections": selections_by_panel, "diagnostics": diagnostics,
        "sources": sources, "legacy_panels": legacy["panels"],
        "future_cells_per_split": {split: sum(row["split"] == split for row in cells) for split in ("ID", "OOD")},
        "future_rmse_wins": {model: {split: sum(row["Time-JEPA"] < row[model] for row in cells if row["split"] == split)
                                     for split in ("ID", "OOD")} for model in ("Static", "Full")},
    }
    (HERE / "vitg_visible_results_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (HERE / "vitg_visible_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "split", "model", "target_count", *METRICS])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in records)
    with (HERE / "vitg_visible_future_per_target.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    lines = [
        "# AgiBot Static / Full / Time-JEPA：同侧手臂可见性汇总",
        "",
        "2026-09-10 按作者要求重新汇总既有结果。本次仅扩展这批 PSG-g 比较的 OOD 汇总口径，不重训，不修改原始预测、指标文件或测试窗口。",
        "",
        TIME_JEPA_NOTICE,
        "",
        "- Readout/Future OOD：左鱼眼只取左臂末端位置 3D、末端速度 3D、夹爪 1D；右鱼眼只取对应右臂 7D。共 14 个视角—同侧 target 项等权，排除底盘平移/转动速度和腰部两维 PCA。",
        "- Shot OOD：左鱼眼取 idx 0/2/4/6，右鱼眼取 idx 1/3/5/7，共 8 项等权；与上一版相同。",
        "- ID 保留三个已见视角的完整面板：Readout/Future 18D，Shot 8D。Future 对 0.2/0.4/0.6/0.8/1.0 s 等权。",
        "",
        "同侧是固定的视角—手臂对应规则，不进行逐帧可见性检测，也不按模型分数挑选变量。ID 与 OOD Readout/Future 的目标集合现在不同，不能把两行差值仅解释为相机域偏移。",
        "",
        "| 任务 | Split | 模型 | Pearson ↑ | R² ↑ | nRMSE ↓ |",
        "|---|---|---|---:|---:|---:|",
    ]
    task_names = {"readout": "Current Readout", "shot": "Current Shot", "future": "Future Readout（五时距）"}
    for row in records:
        peers = [r for r in records if (r["task"], r["split"]) == (row["task"], row["split"])]
        cells_text = []
        for metric in METRICS:
            best = (min if metric == "nrmse" else max)(peer[metric] for peer in peers)
            value = f"{row[metric]:.3f}"
            cells_text.append(f"**{value}**" if row[metric] == best else value)
        lines.append(f"| {task_names[row['task']]} | {row['split']} | {MODEL_LABELS[row['model']]} | " + " | ".join(cells_text) + " |")
    lines += [
        "",
        "ID 三行和 Shot OOD 与上一版一致。Time-JEPA 在三个 ID 面板及 Shot OOD 的三项指标均最优，在全部六个面板的 Pearson 最优。",
        "",
        "Readout OOD 与 Future OOD 的排序发生变化：Static 的 nRMSE 更低且 R² 更高，Time-JEPA 保持较高 Pearson。因此旧口径下‘六面板三项指标均最优’的结论不适用于这张表。",
        "",
        "Full 相对 Static 仍呈现 ID 改善、OOD 退化：新口径下三个 OOD 面板均由 Static 在三项指标上更好。当前结果都是单 encoder/probe seed 点估计。",
        "",
        "原 full-target、cross-side 和共享 4D 的完整分解保存在 `vitg_visible_results_summary.json`；原汇总保留在该 JSON 的 `legacy_panels` 及 `VITG_RESULTS.md`。",
        "",
        "汇总 CSV：`vitg_visible_results.csv`；Future 逐目标 CSV：`vitg_visible_future_per_target.csv`（ID 90 个，OOD 70 个单元）。",
        "",
        "复算：`python external_baselines/psg_jepa/summarize_vitg_visible_results.py`。Law 结果继续使用原 CoVLA 口径。",
    ]
    (HERE / "VITG_VISIBLE_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("PASS: 9 source hashes/histories; legacy macros; same/cross/shared decomposition.")
    print("\n".join(line for line in lines if line.startswith("|")))
    print(json.dumps(payload["future_rmse_wins"]))


if __name__ == "__main__":
    main()
