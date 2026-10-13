"""Audit and summarize the 12-cell, single-domain AgiBot n15000 comparison.

The ten new runs come from the submission manifest. Current Time-JEPA reuses
the two paper-standard artifacts. This does not read or overwrite 7k reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from summarize_vitg_results import baseline_targets, close, finite_tree
from summarize_vitg_visible_results import FISHEYE, side_indices


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "external_baselines/psg_jepa/runs/agibot_n15000_probes_20260912"
PREVIOUS = ROOT / "external_baselines/psg_jepa/runs/agibot_n15000_20260910"
MODELS = ("Static", "Full", "Time-JEPA")
TASKS = ("readout", "shot", "future_readout", "future_shot")
TASK_NAMES = {"readout": "Readout", "shot": "Shot",
              "future_readout": "Future Readout", "future_shot": "Future Shot"}
HORIZONS = ("0.2", "0.4", "0.6", "0.8", "1")
METRICS = ("r2", "pearson", "nrmse")
ALL_METRICS = (*METRICS, "rmse")
TOP = "observation.images.top_head"
ID_VIEWS = {TOP, "observation.images.hand_left", "observation.images.hand_right"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_task(name):
    prefix, task = name.split("_", 1)
    return {"static": "Static", "full": "Full", "time": "Time-JEPA"}[prefix], task


def horizons(task):
    return HORIZONS if task.startswith("future_") else (None,)


def block(data, split, view, horizon):
    value = data["id_view_test" if split == "ID" else "ood_view_test"][view]
    return value if horizon is None else value["per_horizon"][horizon]


def selection(data, task, split, mode="primary"):
    views = data["id_view_test" if split == "ID" else "ood_view_test"]
    all_indices = list(range(len(data["target_fields"])))
    if split == "ID":
        assert set(views) == ID_VIEWS
        selected_views = [TOP] if task.startswith("future_") and mode == "primary" else views
        return {view: all_indices for view in selected_views}
    assert set(views) == set(FISHEYE.values())
    if mode == "full_target":
        return {view: all_indices for view in views}
    if mode == "shared_4d":
        assert task.endswith("readout")
        return {view: [0, 1, 2, 3] for view in views}
    sides = side_indices(data["target_fields"], "shot" if task.endswith("shot") else "readout")
    if mode == "cross_side":
        sides = {"left": sides["right"], "right": sides["left"]}
    return {view: sides[side] for side, view in FISHEYE.items()}


def aggregate(data, split, selected, horizon):
    assert len({len(indices) for indices in selected.values()}) == 1
    values = {
        metric: mean(mean(block(data, split, view, horizon)[metric][i] for i in indices)
                     for view, indices in selected.items())
        for metric in ALL_METRICS
    }
    target_rows = []
    for index in sorted({i for indices in selected.values() for i in indices}):
        records = [block(data, split, view, horizon)
                   for view, indices in selected.items() if index in indices]
        assert records
        assert all(record.get("valid_count", [1] * len(data["targets"]))[index] > 0
                   for record in records)
        target_rows.append({"target_index": index, "target": data["targets"][index],
                            "target_field": data["target_fields"][index],
                            **{metric: mean(row[metric][index] for row in records)
                               for metric in ALL_METRICS}})
    close(values, {metric: mean(row[metric] for row in target_rows) for metric in ALL_METRICS})
    return values, target_rows


def verify_saved_macros(data, task):
    for split, group in (("ID", "id"), ("OOD", "ood")):
        for horizon in horizons(task):
            records = [block(data, split, view, horizon) for view in data[f"{group}_view_test"]]
            saved = data[f"{group}_view_macro_test"]
            saved = saved if horizon is None else saved["per_horizon"][horizon]
            for metric in ALL_METRICS:
                close([mean(column) for column in zip(*(row[metric] for row in records))], saved[metric])
        if task.startswith("future_"):
            saved = data[f"{group}_view_macro_test"]
            for metric in ALL_METRICS:
                close([mean(column) for column in zip(*(saved["per_horizon"][h][metric] for h in HORIZONS))],
                      saved["horizon_macro"][metric])


def verify_same_targets(reference, data, task):
    for field in ("targets", "target_fields", "splits", "future_horizons_seconds"):
        close(reference.get(field), data.get(field))
    for split, group in (("ID", "id"), ("OOD", "ood")):
        assert set(reference[f"{group}_view_test"]) == set(data[f"{group}_view_test"])
        for view in reference[f"{group}_view_test"]:
            for horizon in horizons(task):
                a, b = block(reference, split, view, horizon), block(data, split, view, horizon)
                for item in (a, b):
                    assert all(x > 0 for x in item["nrmse"])
                    assert all(x < 1 for x in item["r2"])
                close([r / n for r, n in zip(a["rmse"], a["nrmse"])],
                      [r / n for r, n in zip(b["rmse"], b["nrmse"])], tolerance=1e-5)
                close([r * r / (1 - s) for r, s in zip(a["rmse"], a["r2"])],
                      [r * r / (1 - s) for r, s in zip(b["rmse"], b["r2"])], tolerance=1e-5)
                if "valid_count" in a:
                    close(a["valid_count"], b["valid_count"])
                if "baselines" in a:
                    close(baseline_targets(a["baselines"]), baseline_targets(b["baselines"]))


def write_csv(name, rows, fields=None):
    fields = fields or list(rows[0])
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def table(panels, tasks, metrics):
    lines = ["| 任务 | 测试 | Static | Full | Time-JEPA 单领域 n15000 |",
             "|---|---|---:|---:|---:|"]
    for task in tasks:
        for split in ("ID", "OOD"):
            peers = {r["model"]: r for r in panels if r["task"] == task and r["split"] == split}
            cells = []
            for model in MODELS:
                cell = []
                for metric in metrics:
                    value = peers[model][metric]
                    best = (min if metric == "nrmse" else max)(r[metric] for r in peers.values())
                    rendered = f"{value:.3f}"
                    cell.append(f"**{rendered}**" if value == best else rendered)
                cells.append(" / ".join(cell))
            lines.append(f"| {TASK_NAMES[task]} | {split} | " + " | ".join(cells) + " |")
    return lines


def main():
    submission = read(OUT / "submission.json")
    previous = read(PREVIOUS / "probe_audit_20260912.json")
    sources = {model_task(r["experiment"]): (Path(r["output_dir"]), r) for r in submission["jobs"]}
    for task in ("readout", "shot"):
        old = previous["original_probe_saved_args_and_metrics"][f"paper_current_{task}"]
        sources["Time-JEPA", task] = (Path(old["path"]).parent, None)
    datasets, audits = {}, []
    for (model, task), (directory, submitted) in sources.items():
        path = directory / "test_metrics.json"
        data, history = read(path), read(directory / "history.json")
        finite_tree(data)
        finite_tree(history)
        assert [row["epoch"] for row in history] == list(range(1, 16))
        best = min(history, key=lambda row: row["val"]["loss"])
        assert best["epoch"] == data["best_epoch"]
        close(best["val"]["loss"], data["best_val"]["loss"])
        assert (directory / "best_probe.pt").is_file()
        assert len(data["targets"]) == (8 if task.endswith("shot") else 18)
        assert data["target_mode"] == ("future_state" if task.startswith("future_") else "state")
        if submitted:
            assert digest(Path(submitted["config"])) == submitted["config_sha256"]
        else:
            old = previous["original_probe_saved_args_and_metrics"][f"paper_current_{task}"]
            assert digest(path) == old["test_metrics_sha256"]
        if model != "Time-JEPA":
            adapter = read(directory / "psg_jepa_probe_manifest.json")["adapter"]
            assert adapter["variant"] == f"psg_{model.lower()}"
            assert adapter["checkpoint"] == submitted["encoder_checkpoint"]
            assert adapter["trainable_encoder_parameters"] == 0
            assert adapter["checkpoint_meta"]["sampling_protocol"] == "standard15000"
            assert adapter["checkpoint_meta"]["embedding_dim"] == 1408
            for key in ("encoder_missing_keys", "encoder_unexpected_keys", "projector_missing_keys", "projector_unexpected_keys"):
                assert not adapter[key]
        verify_saved_macros(data, task)
        datasets[model, task] = data
        audits.append({"model": model, "task": task, "path": str(path), "sha256": digest(path),
                       "history_sha256": digest(directory / "history.json"),
                       "completed_epochs": len(history), "best_epoch": data["best_epoch"],
                       "counts": data["splits"]["counts"], "reused_paper_current": submitted is None})
    for task in TASKS:
        for model in ("Static", "Full"):
            verify_same_targets(datasets["Time-JEPA", task], datasets[model, task], task)

    panels, diagnostic_rows, horizon_rows, target_rows, selections = [], [], [], [], []
    for task in TASKS:
        for split in ("ID", "OOD"):
            chosen = selection(datasets["Time-JEPA", task], task, split)
            selections.append({"task": task, "split": split, "target_indices_by_view": chosen})
            for model in MODELS:
                data = datasets[model, task]
                hvalues, diagnostics = [], {}
                for horizon in horizons(task):
                    values, targets = aggregate(data, split, chosen, horizon)
                    hvalues.append(values)
                    target_rows.extend({"model": model, "task": task, "split": split,
                                        "horizon": float(horizon) if horizon else None, **row} for row in targets)
                    if horizon:
                        horizon_rows.append({"model": model, "task": task, "split": split,
                                             "horizon": float(horizon), **values})
                    modes = ("all_id_views",) if split == "ID" and horizon else ()
                    if split == "OOD":
                        modes = ("full_target", "cross_side") + (("shared_4d",) if task.endswith("readout") else ())
                    for mode in modes:
                        diagnostic, _ = aggregate(data, split, selection(data, task, split, mode), horizon)
                        diagnostics.setdefault(mode, []).append(diagnostic)
                    if split == "OOD":
                        for metric in METRICS:
                            same, cross = values[metric], diagnostics["cross_side"][-1][metric]
                            rebuilt = ((same + cross) / 2 if task.endswith("shot") else
                                       (7 * same + 7 * cross + 4 * diagnostics["shared_4d"][-1][metric]) / 18)
                            close(rebuilt, diagnostics["full_target"][-1][metric], tolerance=1e-12)
                result = {metric: mean(row[metric] for row in hvalues) for metric in ALL_METRICS}
                if task == "shot" and split == "OOD":
                    saved = data["agibot_shot_ood_report"]["primary"]
                    assert saved["name"] == "same_side"
                    close({m: result[m] for m in METRICS}, {m: saved["macro"][m] for m in METRICS})
                panels.append({"task": task, "split": split, "model": model, **result})
                for mode, rows in diagnostics.items():
                    diagnostic_rows.append({"task": task, "split": split, "model": model, "mode": mode,
                                            **{metric: mean(row[metric] for row in rows) for metric in ALL_METRICS}})

    deltas, wins = [], []
    for task in TASKS:
        for split in ("ID", "OOD"):
            peers = {r["model"]: r for r in panels if (r["task"], r["split"]) == (task, split)}
            for first, second in (("Full", "Static"), ("Time-JEPA", "Full"), ("Time-JEPA", "Static")):
                deltas.append({"task": task, "split": split, "comparison": f"{first} minus {second}",
                               **{m: peers[first][m] - peers[second][m] for m in METRICS},
                               "nrmse_relative_reduction_percent": 100 * (1 - peers[first]["nrmse"] / peers[second]["nrmse"])})
            if task.startswith("future_"):
                cells = {(r["model"], r["horizon"], r["target_index"]): r["rmse"] for r in target_rows
                         if (r["task"], r["split"]) == (task, split)}
                indices = [(h, i) for m, h, i in cells if m == "Time-JEPA"]
                for opponent in ("Static", "Full"):
                    wins.append({"task": task, "split": split, "opponent": opponent, "total_cells": len(indices),
                                 "time_jepa_lower_rmse_cells": sum(cells["Time-JEPA", h, i] < cells[opponent, h, i] for h, i in indices)})
    groups = {"robot_speed": [0, 1], "waist": [2, 3], "arm_position": list(range(4, 10)),
              "arm_velocity": list(range(10, 16)), "gripper": [16, 17]}
    group_rows = []
    for task in ("readout", "future_readout"):
        for split in ("ID", "OOD"):
            for group, indices in groups.items():
                for model in MODELS:
                    rows = [r for r in target_rows if r["task"] == task and r["split"] == split
                            and r["model"] == model and r["target_index"] in indices]
                    if rows:
                        group_rows.append({"task": task, "split": split, "group": group, "model": model,
                                           "target_count": len({r["target_index"] for r in rows}),
                                           **{metric: mean(row[metric] for row in rows) for metric in ALL_METRICS}})
    payload = {"report_date": "2026-09-13", "experiment": "single_domain_agibot_n15000",
               "aggregation": {"current_id": "three seen views, equal targets and views",
                               "future_id": "top_head, equal targets and five horizons",
                               "ood": "fixed same-side target sets, equal two fisheye views; future additionally equal five horizons"},
               "sources": audits, "selections": selections, "panels": panels,
               "diagnostics": diagnostic_rows, "per_horizon": horizon_rows,
               "pairwise_deltas": deltas, "future_target_horizon_rmse_wins": wins,
               "readout_target_groups": group_rows,
               "checkpoint_audit": str(OUT / "checkpoint_audit_20260913.json"),
               "audit": "12 complete 15-epoch histories; validation-selected checkpoints; identical targets/counts/scales/target variances; matching available baseline metrics; exact saved macro and same/cross/shared decomposition checks"}
    finite_tree(payload)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv("results.csv", panels)
    write_csv("future_horizons.csv", horizon_rows)
    write_csv("per_target.csv", target_rows)
    write_csv("diagnostics.csv", diagnostic_rows)
    write_csv("pairwise_deltas.csv", deltas)
    write_csv("target_groups.csv", group_rows)

    lines = ["# AgiBot n15000：Static / Full / Time-JEPA 结果", "",
             "2026-09-13，从 10 组新 probe 和复用的 2 组论文 Current 原始 JSON 重新汇总。12 份 history 均完整覆盖 15 epochs，使用各自验证 loss 最低的 probe checkpoint。", "",
             "三模型均为单领域 AgiBot n15000。Time-JEPA 使用论文 e17；Static/Full 使用各自 n15000 最终导出。这里的 Future Time-JEPA 是新做的单领域对照，原论文三领域 Mix Future 和旧 7k 结果均未混入。", "",
             "Current：每格 **R² ↑ / Pearson ↑**。ID 对 top-head、hand-left、hand-right 三视角等权；OOD 为同侧可见目标。", ""]
    lines += table(panels, ("readout", "shot"), ("r2", "pearson"))
    lines += ["", "Future：每格 **R² ↑ / nRMSE ↓**。ID 按原论文取 top-head；0.2/0.4/0.6/0.8/1.0 秒等权。OOD 同样取两鱼眼同侧目标。", ""]
    lines += table(panels, ("future_readout", "future_shot"), ("r2", "nrmse"))
    lines += ["", "Time-JEPA 在全部 8 个面板的两项主指标均最优。Full 相对 Static 的四项 ID R² 均提高，四项 OOD R² 均下降；Current OOD Pearson 提高，但没有转化为更小的预测误差。", "",
              "Readout OOD 每个视角选同侧末端位置 3D、速度 3D、夹爪 1D，共 14 个视角—目标项；Shot 每个视角选同侧 4D，共 8 项。该规则不做逐帧遮挡检测。Current/Future Readout ID 和 OOD 的目标集合不同，不能把二者差值全归为视角迁移。", "",
              "原论文 Current Readout OOD 用全 18D，下面保留该口径方便核对；上表对原 Time-JEPA JSON 同步取了同侧 7D/视角。", "",
              "| Current Readout OOD 口径 | Static R² / Pearson | Full R² / Pearson | Time-JEPA R² / Pearson |",
              "|---|---:|---:|---:|"]
    for mode in ("full_target", "primary"):
        rows = panels if mode == "primary" else diagnostic_rows
        peers = {r["model"]: r for r in rows if r["task"] == "readout" and r["split"] == "OOD" and (mode == "primary" or r["mode"] == mode)}
        label = "全 18D（原论文）" if mode == "full_target" else "同侧 7D/视角（用户要求）"
        lines.append("| " + label + " | " + " | ".join(f"{peers[m]['r2']:.3f} / {peers[m]['pearson']:.3f}" for m in MODELS) + " |")
    lines += ["", "Full 相对 Static 的 ID R² 增量：Current Readout +0.125、Current Shot +0.104、Future Readout +0.085、Future Shot +0.076；对应 OOD 分别下降 0.064、0.050、0.035、0.024。新增动态监督在本次对照中改善了已见视角的可解码性，没有改善同侧鱼眼 OOD 的 R² 或 nRMSE。", "",
              "Time-JEPA 相对 Full 的 ID Future Readout / Shot nRMSE 分别降低 19.5% / 10.0%。五个时距逐项比较，两个 Future 任务在 ID/OOD 的两项主指标均领先；但 Time-JEPA 的 Future Shot OOD 到 1 秒时 R² 为 -0.012，五时距平均 +0.047 不能掩盖这一不足。", "",
              "Current Readout ID 的目标组 R²（同组内目标与三视角等权）：", "",
              "| 目标组 | Static | Full | Time-JEPA |", "|---|---:|---:|---:|"]
    group_labels = {"robot_speed": "底盘速度", "waist": "腰部 PCA", "arm_position": "末端位置",
                    "arm_velocity": "末端速度", "gripper": "夹爪"}
    for group, label in group_labels.items():
        peers = {r["model"]: r for r in group_rows if r["task"] == "readout" and r["split"] == "ID" and r["group"] == group}
        lines.append("| " + label + " | " + " | ".join(f"{peers[m]['r2']:.3f}" for m in MODELS) + " |")
    lines += ["", "Full 对 Static 的 Current 末端速度 R² 从 0.133 提高到 0.314，与其增加速度监督的方向一致；Time-JEPA 为 0.764。Future Readout 的末端速度 R² 为 Static 0.088、Full 0.155、Time-JEPA 0.207，差距小于腰部和位置组；不能把 Future 宏平均优势都解释为速度预测改善。", ""]
    lines += ["", "本次保留同一组数据与 probe 设置下的方法差异：PSG 为逐帧图像 encoder + projector，Time-JEPA 为视频 encoder。Future 最后一个 PSG 特征来自最后两张独立图像，Time-JEPA 对应特征含观测片段时间上下文；结果不能仅归因于辅助 loss。三者训练计划同为 20 epochs，所选 encoder 为 PSG e19 / Time e17；目前只有一个 encoder/probe seed，未做显著性检验。", "",
              "逐 target × horizon 的 RMSE 胜数用于区分宏平均优势与逐变量优势，不作为独立实验重复次数：", "",
              "| 任务 | 测试 | Time-JEPA 胜 Static | Time-JEPA 胜 Full |",
              "|---|---|---:|---:|"]
    for task in ("future_readout", "future_shot"):
        for split in ("ID", "OOD"):
            peers = {r["opponent"]: r for r in wins if r["task"] == task and r["split"] == split}
            lines.append(f"| {TASK_NAMES[task]} | {split} | " + " | ".join(f"{peers[m]['time_jepa_lower_rmse_cells']}/{peers[m]['total_cells']}" for m in ("Static", "Full")) + " |")
    lines += ["", "[完整数值与审计来源](results.json) · [主表 CSV](results.csv) · [逐时距](future_horizons.csv) · [逐变量](per_target.csv) · [全维/对侧/共享变量及 Future 三视角诊断](diagnostics.csv) · [模型差值](pairwise_deltas.csv) · [目标组](target_groups.csv) · [12 份 checkpoint 参数与 optimizer step 核查](checkpoint_audit_20260913.json)", "",
              "复算命令：`python external_baselines/psg_jepa/summarize_vitg_n15000_results.py`。本脚本只读取既有 probe 指标，不改动 checkpoint 或原始结果。", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print("Audited 12 probe artifacts and wrote", OUT / "RESULTS.md")
    for row in panels:
        print(row["task"], row["split"], row["model"], *(f"{row[m]:.6f}" for m in METRICS))
    print("Future target/horizon RMSE wins:", json.dumps(wins))


if __name__ == "__main__":
    main()
