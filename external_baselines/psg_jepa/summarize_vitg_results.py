"""Recompute the completed PSG-g comparison from saved probe metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = Path(
    ""
    "outputs_external/psg_jepa_vitg"
)
MODELS = {
    "Static": "agibot_static_g_port_seed239",
    "Full": "agibot_full_g_port_seed239",
    "Time-JEPA": "time_jepa_g_seed239",
}
METRICS = ("pearson", "r2", "nrmse")
HORIZONS = ("0.2", "0.4", "0.6", "0.8", "1")
MODEL_LABELS = {"Static": "Static", "Full": "Full", "Time-JEPA": "Time-JEPA（7k 对照）"}
TIME_JEPA_NOTICE = (
    "**实验身份更正（2026-09-10）：本页 AgiBot 的 Time-JEPA 是 quota7000/e19、"
    "probe-train 1500 的对照实验。论文 Current 主表使用 n15000/e17、probe-train 4500；"
    "论文 Future 使用三领域 Mix-Exposure/e19，ID 取 top-head，而本批 Future ID 是三视角平均。"
    "这些是不同实验的结果，不能互相替代。** 完整数值对照与原始来源见 "
    "[TIME_JEPA_EXPERIMENT_SCOPE.md](TIME_JEPA_EXPERIMENT_SCOPE.md)。"
)


def read_json(path):
    return json.loads(path.read_text())


def finite_tree(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def close(a, b, tolerance=1e-6):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            close(a[key], b[key], tolerance)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y, tolerance)
    elif isinstance(a, (float, int)):
        assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)
    else:
        assert a == b


def macro(values):
    return {metric: mean(values[metric]) for metric in METRICS}


def baseline_targets(values):
    return {name: {key: item[key] for key in ("mae", "rmse", "nrmse", "valid_count")}
            for name, item in values.items()}


def view_arrays(data, split, horizon=None):
    views = data["id_view_test" if split == "ID" else "ood_view_test"]
    records = [value if horizon is None else value["per_horizon"][horizon]
               for value in views.values()]
    arrays = {
        metric: [mean(column) for column in zip(*(record[metric] for record in records))]
        for metric in (*METRICS, "rmse")
    }
    saved = data["id_view_macro_test" if split == "ID" else "ood_view_macro_test"]
    if horizon is not None:
        saved = saved["per_horizon"][horizon]
    for metric, values in arrays.items():
        close(values, saved[metric])
    return arrays


def load_probe(model, task):
    if model == "Time-JEPA" and task != "future":
        directory = ROOT / "time-jepa/outputs_covla_agibot_egoexo/agibot_only/agibot" / task / "conv1x1"
    else:
        folder = ("future_state_1s_5step_readout_seed0" if task == "future"
                  else f"{task}_conv1x1_seed0")
        directory = RUNS / "probes" / MODELS[model] / folder
    path = directory / "test_metrics.json"
    data = read_json(path)
    history = read_json(directory / "history.json")
    finite_tree(data)
    finite_tree(history)
    assert [item["epoch"] for item in history] == list(range(1, 16))
    selected = min(history, key=lambda item: item["val"]["loss"])
    assert selected["epoch"] == data["best_epoch"]
    close(selected["val"]["loss"], data["best_val"]["loss"])
    assert (directory / "best_probe.pt").is_file()
    if model != "Time-JEPA":
        adapter = read_json(directory / "psg_jepa_probe_manifest.json")["adapter"]
        assert adapter["variant"] == f"psg_{model.lower()}"
        assert adapter["trainable_encoder_parameters"] == 0
        for key in ("encoder_missing_keys", "encoder_unexpected_keys",
                    "projector_missing_keys", "projector_unexpected_keys"):
            assert not adapter[key]
    audit = {
        "model": model, "task": task, "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "completed_epochs": len(history), "best_epoch": data["best_epoch"],
        "counts": data["splits"]["counts"],
    }
    return data, audit


def main():
    datasets, audits = {}, []
    for task in ("readout", "shot", "future"):
        for model in MODELS:
            data, audit = load_probe(model, task)
            datasets[model, task] = data
            audits.append(audit)
        reference = datasets["Time-JEPA", task]
        for model in ("Static", "Full"):
            data = datasets[model, task]
            for key in ("target_fields", "targets", "splits", "future_horizons_seconds"):
                close(data.get(key), reference.get(key))
            for view_group in ("id_view_test", "ood_view_test"):
                assert set(data[view_group]) == set(reference[view_group])
                for view, values in data[view_group].items():
                    comparison = reference[view_group][view]
                    # Older Time-JEPA current-state JSON predates these fields.
                    # Its normalization scales and target variance remain
                    # recoverable from RMSE/nRMSE and RMSE^2/(1-R2).
                    for key in ("valid_count", "baselines"):
                        if key in comparison:
                            if key == "baselines":
                                close(baseline_targets(values[key]), baseline_targets(comparison[key]))
                            else:
                                close(values[key], comparison[key])
                    if task != "future":
                        for record in (values, comparison):
                            assert all(x > 0 for x in record["nrmse"])
                            assert all(x < 1 for x in record["r2"])
                        close([r / n for r, n in zip(values["rmse"], values["nrmse"])],
                              [r / n for r, n in zip(comparison["rmse"], comparison["nrmse"])])
                        close([r * r / (1 - s) for r, s in zip(values["rmse"], values["r2"])],
                              [r * r / (1 - s) for r, s in zip(comparison["rmse"], comparison["r2"])],
                              tolerance=1e-5)
                    if task == "future":
                        for horizon in HORIZONS:
                            close(values["per_horizon"][horizon]["valid_count"],
                                  comparison["per_horizon"][horizon]["valid_count"])
                            close(baseline_targets(values["per_horizon"][horizon]["baselines"]),
                                  baseline_targets(comparison["per_horizon"][horizon]["baselines"]))

    panels, diagnostics, per_horizon, cells, future_groups = [], [], [], [], []
    for task in ("readout", "shot", "future"):
        for split in ("ID", "OOD"):
            row = {"task": task, "split": split, "models": {}}
            all_arrays = {}
            for model in MODELS:
                data = datasets[model, task]
                if task == "future":
                    arrays = [view_arrays(data, split, horizon) for horizon in HORIZONS]
                    values = {metric: [mean(column) for column in zip(*(a[metric] for a in arrays))]
                              for metric in METRICS}
                    saved = data["id_view_macro_test" if split == "ID" else "ood_view_macro_test"]["horizon_macro"]
                    for metric in METRICS:
                        close(values[metric], saved[metric])
                    for name, indices in {
                        "motion_8d": [0, 1, 10, 11, 12, 13, 14, 15],
                        "configuration_10d": [2, 3, 4, 5, 6, 7, 8, 9, 16, 17],
                    }.items():
                        future_groups.append({
                            "model": model, "split": split, "group": name,
                            "target_fields": [data["target_fields"][index] for index in indices],
                            **{metric: mean(values[metric][index] for index in indices)
                               for metric in METRICS},
                        })
                    for horizon, values_at_h in zip(HORIZONS, arrays):
                        per_horizon.append({"model": model, "split": split,
                                            "horizon": float(horizon), **macro(values_at_h)})
                        all_arrays[model, horizon] = values_at_h
                else:
                    values = view_arrays(data, split)
                    if task == "shot" and split == "OOD":
                        report = data["agibot_shot_ood_report"]
                        assert report["primary"]["name"] == "same_side"
                        primary = report["primary"]
                        values = {metric: [mean(view["metrics"][metric])
                                           for view in primary["views"].values()]
                                  for metric in METRICS}
                        close(macro(values), {m: primary["macro"][m] for m in METRICS})
                        for name, diagnostic in report["diagnostics"].items():
                            diagnostics.append({"model": model, "scope": name,
                                                **{m: diagnostic["macro"][m] for m in METRICS}})
                row["models"][model] = macro(values)
            if task == "future":
                for horizon in HORIZONS:
                    for index, target in enumerate(datasets["Time-JEPA", task]["targets"]):
                        cells.append({"split": split, "horizon": float(horizon), "target": target,
                                      **{model: all_arrays[model, horizon]["rmse"][index]
                                         for model in MODELS}})
            panels.append(row)

    training = []
    for model in ("Static", "Full"):
        directory = RUNS / MODELS[model]
        files = list((directory / "csv").rglob("metrics.csv"))
        assert len(files) == 1
        with files[0].open() as handle:
            epochs = [row for row in csv.DictReader(handle) if row.get("train/loss_epoch")]
        assert [int(float(row["epoch"])) for row in epochs] == list(range(20))
        assert int(float(epochs[-1]["step"])) == 2179
        final = {key: float(value) for key, value in epochs[-1].items() if value}
        finite_tree(final)
        training.append({"model": model, "completed_epochs": 20,
                         "scheduled_updates": 2180, "last_metrics": final})

    summary = {
        "status": "METRIC_AND_HISTORY_AUDIT_PASS", "encoder_seed": 239, "probe_seed": 0,
        "model_labels": MODEL_LABELS,
        "time_jepa_experiment_scope": "AgiBot quota7000 e19, probe-train 1500; distinct from paper Current n15000 e17 / probe-train 4500 and paper Future Mix-Exposure e19 / top-head ID",
        "aggregation": "equal view then equal target; future also equal horizon; Shot OOD same-side",
        "audit_scope": "JSON/history, saved loading manifests and training CSV; future baselines/valid counts match; legacy Time current JSON lacks baselines/valid counts, so check split counts and recover normalization/target variance from metrics; no new checkpoint deserialization or cluster query",
        "inputs": audits, "training": training, "panels": panels,
        "shot_ood_diagnostics": diagnostics, "per_horizon": per_horizon,
        "future_groups": future_groups,
        "future_group_analysis_scope": "post-hoc analysis using the existing PSG velocity/configuration target partition",
        "future_rmse_wins": {
            model: {split: sum(cell["Time-JEPA"] < cell[model] for cell in cells if cell["split"] == split)
                    for split in ("ID", "OOD")}
            for model in ("Static", "Full")
        },
        "future_cells_per_split": 90,
    }
    (HERE / "vitg_results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (HERE / "vitg_future_per_target.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    lines = [
        "# PSG 单点状态监督与 Law 监督：完成结果分析",
        "",
        "日期：2026-09-09。PSG 数字由本脚本从正式 JSON 重算；Law 数字来自已完成审计、并再次对原始 JSON 与 SHA256 核验的单领域比较。",
        "",
        "2026-09-10：AgiBot Readout/Future OOD 已按同侧手臂重新汇总，最新表见 [VITG_VISIBLE_RESULTS.md](VITG_VISIBLE_RESULTS.md)。本页保留原全维度 Readout/Future OOD 结果及方法说明，Law 结果继续适用。",
        "",
        TIME_JEPA_NOTICE,
        "",
        "## 1. 两种监督具体怎样做",
        "",
        "### 单点额外监督：PSG-Static-g-Port，以及 PSG-Full-g-Port",
        "",
        "这里的单点是 single-latent grounding：每帧独立编码，再从该帧表示回归当时的状态。它在每个采样帧上都有监督，并非每段视频只标一个时刻。",
        "",
        "- Static：`x_t → ViT-g image branch → spatial mean → projector → z_t → H_s → s_t`，以 masked MSE 回归归一化后的 18D 状态。",
        "- Full：保留 Static，并用两个 latent 预测 10D 配置变化及 8D 运动量。配置变化包括腰部两维 PCA、双臂末端 6D 位置和两个夹爪；运动量包括底盘平移/转动速度与双臂末端 6D 速度。",
        "- Static 总损失为 `L_forward + 0.09 L_SIGReg + 0.1 L_static`；Full 为 `L_forward + 0.09 L_SIGReg + 0.1 (L_static + L_delta + L_velocity)`。Full 的 delta 监督覆盖四帧序列内的 0.2/0.4/0.6 s 间隔；velocity head 使用相邻 latent 对预测左端时刻的运动量。它们是标签回归，不是显式物理方程残差。",
        "- 每个 64f@5FPS 观测窗口拆成 16 组四帧序列。训练保留 PSG 的 action-conditioned 自回归 predictor；每次 5Hz 转移对应六条 30Hz、40D command 拼成的 240D action。",
        "",
        "两个 PSG 变体与本表 Time-JEPA 共用 released V-JEPA 2.1 ViT-g checkpoint、384px、7,000 个 view-windows（3000/2000/2000）、18D 状态语义、20 epochs 和 global window batch 64，encoder seed 为 239。PSG 的 projector/predictor/heads 随机初始化。",
        "",
        "评估时移除训练辅助头。PSG 保留冻结的逐帧 encoder+projector，把相邻两帧特征平均为 32 个位置，另训同样的 Conv1x1 probe。Time-JEPA 的逐位置回归也使用 pointwise head，但输入特征已经由整个视频片段的时空注意力产生。",
        "",
        "### Law：Time-JEPA + 显式纵向运动学约束",
        "",
        "在 CoVLA 上保留原 `vEgo/aEgo/yaw_rate` 状态监督，用同一个状态头的预测增加：",
        "",
        "```text",
        "r[k] = v_hat[k+1] - v_hat[k] - dt[k] * (a_hat[k] + a_hat[k+1]) / 2",
        "L_law = mean_valid SmoothL1(r / s_delta_v, beta=0.5)",
        "L_total = L_JEPA + lambda(step) * (L_state + 0.5 * L_law)",
        "```",
        "",
        "速度和加速度先还原到 m/s 与 m/s²；`dt` 用映射帧的视频 PTS 计算 tubelet 中心时间差。两个 tubelet 的四帧都必须具有有效 v/a，时间有序，且 `0 < dt <= 0.6 s`。尺度 `s_delta_v=0.2612949457 m/s` 只从 encoder-train 得到。lambda 在 250 steps 内升到 0.2，因此最终 Law 权重为 0.1。",
        "",
        "该条件没有新增参数或额外 held-out 标签。它保留真实 state loss 作为锚点，避免常速度、零加速度也能满足方程的退化解。alpha=0.5 在训练前固定。本次比较使用相同单领域、6,305-video strict 协议的 Time-JEPA e19；两边均重新训练普通冻结 probes，probe 阶段不加 Law loss。",
        "",
        "## 2. 单点监督结果：AgiBot",
        "",
        "ID 为三个已见视角等权平均，OOD 为两个 held-out fisheye 视角。Readout 是 18D 训练监督量；Shot 是 8D encoder 未监督量，但 probe 仍有监督标签。Shot OOD 使用 same-side 主口径。本表已完成的 Future 只包含这 18D Readout 量，对 0.2/0.4/0.6/0.8/1.0 s 等权；缺少 Future Shot 仅指本批 7k 比较，论文 Mix Future Shot 已有完成结果。",
        "",
        "当前 probe train/val 为 1500/1500，每个 test view 1000；Future 有效样本为 1479/1476，每个 test view 989。三模型的相应计数一致；所有 probes 均完成 15 epochs、seed 0，按 validation loss 选点。",
        "",
        "| 任务 | Split | 模型 | Pearson ↑ | R² ↑ | nRMSE ↓ |",
        "|---|---|---|---:|---:|---:|",
    ]
    task_names = {"readout": "Current Readout", "shot": "Current Shot", "future": "Future Readout（五时距）"}
    for row in panels:
        for model, values in row["models"].items():
            lines.append(f"| {task_names[row['task']]} | {row['split']} | {MODEL_LABELS[model]} | "
                         + " | ".join(f"{values[metric]:.3f}" for metric in METRICS) + " |")
    lines += [
        "",
        "Time-JEPA 在六个主面板的三项宏平均上全部最优。相对 Static，其 ID Readout/Shot/Future nRMSE 分别下降 18.010%/17.459%/9.778%；相对 Full 分别下降 14.314%/12.306%/7.741%。",
        "",
        "Full 对 Static 的变化呈现 ID/OOD 分化：三个 ID 任务的 nRMSE 下降 4.313%/5.875%/2.208%，对应三个 OOD 任务却上升 3.507%/5.323%/2.754%。这说明增加成对状态变化监督改善了已见视角的表示可读性，但没有在本次运行中转化为鱼眼视角泛化。",
        "",
        "Time-JEPA 的 OOD 优势也有限：Readout/Future 的 R² 仍为 -0.015/-0.034，尽管 Pearson 达到 0.298/0.223。与 Static 相比这两项 nRMSE 仅下降 1.405%/0.864%；相关性更高不等于误差校准良好。",
        "",
        "### Future 的配置量与运动量拆分",
        "",
        "这是看到结果后的解释性分组，沿用配置中已有的 10D configuration / 8D motion 划分，不作预注册声明。每组仍为 view、horizon、target 等权平均。",
        "",
        "| 分组 | Split | 模型 | Pearson ↑ | R² ↑ | nRMSE ↓ |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in future_groups:
        lines.append(f"| {row['group']} | {row['split']} | {row['model']} | "
                     + " | ".join(f"{row[metric]:.3f}" for metric in METRICS) + " |")
    lines += [
        "",
        "Time-JEPA 的 Future 总体优势主要来自配置量。8D 运动量上，ID 的 Full nRMSE/R² 为 0.947/0.046，优于 Time-JEPA 的 0.965/0.010；OOD 运动量的 Static、Full nRMSE 也低于 Time-JEPA。",
        "",
        f"按 view-macro 后的 target×horizon 单元计算，Time-JEPA 对 Static 在 ID/OOD 分别胜 {summary['future_rmse_wins']['Static']['ID']}/90、{summary['future_rmse_wins']['Static']['OOD']}/90，对 Full 为 {summary['future_rmse_wins']['Full']['ID']}/90、{summary['future_rmse_wins']['Full']['OOD']}/90。因此不能把六个宏平均面板最优改写为所有运动量或所有时距均最优。",
        "",
        "比较解释还受到特征时间范围的限制：Future probe 只读最后一个位置，PSG 该位置仅是最后两帧的独立特征平均；Time-JEPA 该位置已包含全片段上下文。PSG 的训练期 AR predictor 在这里没有进入 frozen feature path，因此这张表衡量冻结表示的未来状态可解码性，不衡量 PSG 原 predictor 的 action-conditioned rollout 能力。",
        "",
        "这批实验控制了 backbone 规模、预训练来源、状态语义和视频曝光预算，但同时存在图像/视频 stem、时间上下文、训练目标和 action conditioning 的差异；它是方法移植比较，不能单独归因于监督是否逐点。该 ViT-g 批次也没有无 grounding 的 PSG-g 对照。全部差异是单 encoder/probe seed 的点估计。",
        "",
        "## 3. Law 结果：CoVLA",
        "",
        "Current Readout/Future Seen 是 vEgo、aEgo、yaw_rate；Current Shot/Future Unseen 是方向盘角、横向速度、横向加速度。OOD 为雨天。Future 对五个片外时距等权。两边均为单领域 Time-JEPA e19、encoder/probe seed 239。",
        "",
        "| 任务 | Split | Time nRMSE ↓ | +Law nRMSE ↓ | Law 误差变化 | Time R² ↑ | +Law R² ↑ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    law_path = ROOT / "time-jepa/outputs_law_comparison/covla_alpha0p5_e19_seed239/results_summary.json"
    law = read_json(law_path)
    law_hashes = {row["test_metrics"]: row["test_metrics_sha256"]
                  for row in law["stage_audits"]}
    for item in law["inputs"]:
        law_hashes[item["time_metrics"]] = item["time_sha256"]
        for model in ("time", "law"):
            path = Path(item[f"{model}_metrics"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == law_hashes[str(path)]
            data = read_json(path)
            for split in ("test", "ood_test"):
                values = data[split]
                calculated = ({metric: mean(mean(h[metric]) for h in values["per_horizon"].values())
                               for metric in METRICS}
                              if item["task"].startswith("future") else macro(values))
                matching = [row for row in law["panels"]
                            if row["split"] == split and f"{row['mode']}_{row['group']}" == item["task"]]
                assert len(matching) == 1
                close(calculated, matching[0][model])
    for row in law["panels"]:
        name = ("Current Readout" if row["group"] == "readout" else "Current Shot") if row["mode"] == "current" else ("Future Seen" if row["group"] == "readout" else "Future Unseen")
        split = "ID" if row["split"] == "test" else "OOD"
        lines.append(f"| {name} | {split} | {row['time']['nrmse']:.3f} | {row['law']['nrmse']:.3f} | {row['law_nrmse_change_pct']:+.3f}% | {row['time']['r2']:.3f} | {row['law']['r2']:.3f} |")
    lines += [
        "",
        "Law 在 8/8 个面板的 nRMSE、R²、Pearson 上均落后；Current 的 12/12 个 target×split，Future 的 57/60 个 target×horizon×split 的 RMSE 增大。只有三个 OOD future cells 小幅改善，幅度均小于 0.3%。",
        "",
        "退化最明显的是当前 Readout（约 11%–12%）。直接参与规律的 aEgo 在 ID/OOD 上 RMSE 分别增加 12.472%/12.251%，未参与方程但同样受状态监督的 yaw_rate 增加 15.424%/13.684%。因此当前证据没有显示‘牺牲其它变量，换取 law variables 读出更准’。",
        "",
        "一个有数据依据、但尚未得到模型层面验证的解释是：标签自己的方程残差非零，同步状态损失与理想化规律损失可能存在冲突。train-only gate 的梯形残差 MAE 为 0.0506484 m/s，约为实际速度变化 MAE 的 30.1%；将加速度移动一个 tubelet 的诊断残差更低，但正式训练保留原标签时间对齐。",
        "",
        "结果支持的结论是：在本次 alpha=0.5、相同训练预算下，增加这条显式纵向运动学正则没有改善状态读出和未来预测。训练日志与梯度检查确认该项参与了优化，但本批没有两模型独立的规律残差或轨迹重建评估，不能认定其规律满足程度是否改善，也不能推广到所有显式物理方法。六个新增 Law/Time probes 均完成 15 epochs 且 best 都在末轮，结论限定为固定预算的点估计。",
        "",
        "## 4. 复算与完整产物",
        "",
        "- PSG 重算：`python external_baselines/psg_jepa/summarize_vitg_results.py`。",
        "- PSG 机器汇总、逐 horizon 和 source hashes：`external_baselines/psg_jepa/vitg_results_summary.json`。",
        "- PSG 全部 180 个 future 单元（指标为 view-macro RMSE）：`external_baselines/psg_jepa/vitg_future_per_target.csv`。",
        "- PSG 原始结果根目录：`outputs_external/psg_jepa_vitg/probes/`。",
        "- Law 完整表与逐时距：`time-jepa/outputs_law_comparison/covla_alpha0p5_e19_seed239/RESULTS.md`。",
        "- Law 全部 72 个单元：同目录 `per_target_results.csv`；完成审计：`completion_summary.json`。",
        "",
        "本次 PSG 核验覆盖九份 metric/history、两份 encoder 20-epoch CSV、已保存的 encoder/projector 加载 manifest、样本数、view/target/horizon 宏平均。Future 的 GT baseline 和 valid counts 跨模型相同；较早 Time current JSON 未存 baseline/valid counts，改以相同 split counts 和 RMSE/nRMSE、RMSE²/(1-R²) 恢复的 normalization/target variance 核验。没有重新加载大型 encoder checkpoint 或重新查询集群状态。",
    ]
    (HERE / "VITG_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(summary["status"])
    print("9 probe histories, 2 encoder histories, 6 PSG panels; report includes 8 Law panels.")
    print(json.dumps(summary["future_rmse_wins"]))


if __name__ == "__main__":
    main()
