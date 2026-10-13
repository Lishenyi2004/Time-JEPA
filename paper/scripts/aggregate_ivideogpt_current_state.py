"""Recompute the main-text iVideoGPT current-state aggregates from probe JSON."""

from pathlib import Path
import json
from statistics import mean


ROOT = Path(__file__).resolve().parents[2] / "time-jepa/outputs_world_model/ivideogpt_oxe_64_medium"
MODES = ("frozen", "video_only", "time")
GROUPS = ("readout", "shot")
SPLITS = ("id", "ood")
DOMAINS = {
    "autonomous driving": ("covla/weather_ood", 3, 3),
    "robot manipulation": ("agibot_n15000", 18, 8),
    "human motion": ("egoexo/exo124_view3ood", 9, 5),
}


def result_path(domain: str, mode: str, group: str) -> Path:
    stem = DOMAINS[domain][0]
    if domain == "robot manipulation":
        if mode == "frozen":
            stem = "agibot_wholebody_v2"
        return ROOT / mode / stem / group / "probe_3view/conv1x1/test_metrics.json"
    if domain == "autonomous driving":
        return ROOT / mode / stem / group / "conv1x1_seed239/test_metrics.json"
    return ROOT / mode / stem / group / "conv1x1/test_metrics.json"


def score(data: dict, domain: str, group: str, split: str, metric: str) -> float:
    if domain == "robot manipulation" and group == "shot" and split == "ood":
        return data["agibot_shot_ood_report"]["primary"]["macro"][metric]
    if domain == "autonomous driving":
        key = "test" if split == "id" else "ood_test"
    else:
        key = f"{split}_view_macro_test"
    return mean(data[key][metric])


def main() -> None:
    for split in SPLITS:
        for group_index, group in enumerate(GROUPS):
            weights = [values[group_index + 1] for values in DOMAINS.values()]
            for metric in ("pearson", "r2"):
                values = []
                for mode in MODES:
                    domain_scores = []
                    for domain in DOMAINS:
                        data = json.loads(result_path(domain, mode, group).read_text())
                        domain_scores.append(score(data, domain, group, split, metric))
                    values.append(sum(w * x for w, x in zip(weights, domain_scores)) / sum(weights))
                print(f"{split:3} {group:7} {metric:7} " + " ".join(f"{v:.6f}" for v in values))


if __name__ == "__main__":
    main()
