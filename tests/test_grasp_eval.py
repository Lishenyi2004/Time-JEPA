from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from grasp_eval_rjob.compare_models import _accuracy_rows, _stratified_delta_bootstrap
from grasp_eval_rjob.config import validate_config
from grasp_eval_rjob.constants import FAMILY_TO_PROPERTIES, FAMILIES, PROPERTIES
from grasp_eval_rjob.dataset import discover_pairs
from grasp_eval_rjob.run_eval import evaluate_pair
from grasp_eval_rjob.score import summarize, video_score_table


class _FakeSurpriseModel:
    tubelet_size = 2

    def encode_target(self, clips: torch.Tensor) -> torch.Tensor:
        return torch.zeros((clips.shape[0], 1, 1), device=clips.device)

    def surprise_from_target(
        self, clips: torch.Tensor, target: torch.Tensor, context: int
    ) -> torch.Tensor:
        del target
        return clips[:, 0, 0, 0, 0].float() + context / 100.0


def _minimal_config() -> dict:
    return {
        "name": "test",
        "checkpoint": "/tmp/checkpoint.pt",
        "checkpoint_sha256": "0" * 64,
        "dataset_root": "/tmp/grasp",
        "data_audit": "/tmp/data_audit.json",
        "output_dir": "/tmp/output",
        "model": {
            "resolution": 384,
            "patch_size": 16,
            "tubelet_size": 2,
            "pred_depth": 24,
            "num_mask_tokens": 2,
            "levels_predictor": 4,
            "weights_dtype": "float32",
        },
        "evaluation": {
            "frames_per_clip": 16,
            "frame_step": 10,
            "stride": 2,
            "context_frames": 2,
            "chunk_size": 1,
            "num_workers": 0,
            "max_pairs": None,
        },
        "scoring": {
            "minimum_nondegraded_properties": 6,
            "require_leave_largest_gain_out_positive": True,
        },
    }


class GRASPEvalTests(unittest.TestCase):
    def test_mapping_has_eight_nonempty_properties_and_sixteen_families(self) -> None:
        self.assertEqual(len(FAMILIES), 16)
        self.assertEqual(len(PROPERTIES), 8)
        mapped = {prop for values in FAMILY_TO_PROPERTIES.values() for prop in values}
        self.assertEqual(mapped, set(PROPERTIES))
        self.assertEqual(FAMILY_TO_PROPERTIES["GravitySupport"], ("Gravity", "Support"))
        self.assertEqual(
            FAMILY_TO_PROPERTIES["Unchangeableness"], ("Color Constancy",)
        )

    def test_config_rejects_context_sweep_or_wrong_nondegradation_threshold(self) -> None:
        config = _minimal_config()
        validate_config(config)
        config["evaluation"]["context_frames"] = 4
        with self.assertRaisesRegex(ValueError, "Fixed GRASP protocol"):
            validate_config(config)
        config = _minimal_config()
        config["scoring"]["minimum_nondegraded_properties"] = 7
        with self.assertRaisesRegex(ValueError, "frozen at 6/8"):
            validate_config(config)

    def test_pair_discovery_requires_exact_p_ip_filename_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in FAMILIES:
                for prefix in ("P", "IP"):
                    directory = root / f"{prefix}_{family}"
                    directory.mkdir()
                    (directory / "0.mp4").touch()
            pairs = discover_pairs(root, require_full=False)
            self.assertEqual(len(pairs), len(FAMILIES))
            self.assertEqual(pairs[0].pair_id, "Collision:0")
            (root / "IP_Gravity" / "0.mp4").rename(
                root / "IP_Gravity" / "1.mp4"
            )
            with self.assertRaisesRegex(RuntimeError, "Pair filenames differ"):
                discover_pairs(root, require_full=False)

    def test_window_order_and_average_surprise(self) -> None:
        clips = torch.zeros(2, 3, 20, 2, 2)
        for video in range(2):
            for frame in range(20):
                clips[video, :, frame] = video * 100 + frame
        output = evaluate_pair(
            model=_FakeSurpriseModel(),
            clips=clips,
            evaluation={
                "frames_per_clip": 16,
                "stride": 2,
                "context_frames": 2,
                "chunk_size": 2,
                "use_bfloat16": False,
            },
            device=torch.device("cpu"),
        )
        self.assertEqual(output["window_start"].tolist(), [0, 2, 4])
        self.assertEqual(output["prediction_start"].tolist(), [2, 4, 6])
        np.testing.assert_allclose(
            output["possible_losses"].numpy(), [0.02, 2.02, 4.02], rtol=0, atol=1e-5
        )
        np.testing.assert_allclose(
            output["impossible_losses"].numpy(),
            [100.02, 102.02, 104.02],
            rtol=0,
            atol=1e-5,
        )

    def test_strict_ties_are_errors_and_property_macro_is_equal_weighted(self) -> None:
        records = [
            {
                "dataset_index": 0,
                "pair_id": "Collision:0",
                "family": "Collision",
                "file_name": "0.mp4",
                "properties": ["Collision"],
                "possible_relative_path": "P_Collision/0.mp4",
                "impossible_relative_path": "IP_Collision/0.mp4",
                "window_start": torch.tensor([0, 2]),
                "prediction_start": torch.tensor([2, 4]),
                "possible_losses": torch.tensor([1.0, 1.0]),
                "impossible_losses": torch.tensor([1.0, 1.0]),
            },
            {
                "dataset_index": 1,
                "pair_id": "GravitySupport:0",
                "family": "GravitySupport",
                "file_name": "0.mp4",
                "properties": ["Gravity", "Support"],
                "possible_relative_path": "P_GravitySupport/0.mp4",
                "impossible_relative_path": "IP_GravitySupport/0.mp4",
                "window_start": torch.tensor([0, 2]),
                "prediction_start": torch.tensor([2, 4]),
                "possible_losses": torch.tensor([1.0, 1.0]),
                "impossible_losses": torch.tensor([2.0, 2.0]),
            },
        ]
        windows, _, pairs = video_score_table(records, context_frames=2)
        self.assertEqual(len(windows), 8)
        average = pairs[pairs["aggregation"] == "average"].set_index("pair_id")
        self.assertEqual(average.loc["Collision:0", "correct"], 0)
        self.assertEqual(average.loc["GravitySupport:0", "correct"], 1)

    def test_primary_rows_use_equal_property_macro(self) -> None:
        rows = []
        for index, family in enumerate(FAMILIES):
            rows.append(
                {
                    "pair_id": f"{family}:0",
                    "family": family,
                    "properties": "|".join(FAMILY_TO_PROPERTIES[family]),
                    "correct::model": int(family == "GravitySupport"),
                }
            )
        frame = pd.DataFrame(rows)
        metrics = pd.DataFrame(_accuracy_rows(frame, "model"))
        property_values = metrics.loc[
            metrics["group_type"] == "property", "accuracy"
        ].to_numpy()
        macro = metrics.loc[
            metrics["group_type"] == "property_macro", "accuracy"
        ].item()
        self.assertAlmostEqual(macro, float(property_values.mean()))
        self.assertNotEqual(
            macro,
            metrics.loc[metrics["group_type"] == "overall_micro", "accuracy"].item(),
        )

    def test_paired_bootstrap_is_zero_for_identical_models(self) -> None:
        rows = []
        for family in FAMILIES:
            for index in range(3):
                correct = (index + len(family)) % 2
                rows.append(
                    {
                        "pair_id": f"{family}:{index}",
                        "family": family,
                        "correct::time": correct,
                        "correct::video": correct,
                    }
                )
        output = _stratified_delta_bootstrap(
            pd.DataFrame(rows),
            time_column="correct::time",
            video_column="correct::video",
            samples=100,
            seed=239,
        )
        self.assertEqual(output["property_macro_ci95_pp"], [0.0, 0.0])
        self.assertEqual(output["overall_micro_ci95_pp"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
