import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))

from probe_view_metrics import (  # noqa: E402
    build_agibot_shot_ood_report,
    validate_agibot_same_side_ood,
)


TARGET_FIELDS = [
    "state/joint/position_norm_left",
    "state/joint/position_norm_right",
    "state/joint/velocity_norm_left",
    "state/joint/velocity_norm_right",
    "state/joint/effort_norm_left",
    "state/joint/effort_norm_right",
    "state/end/wrench_force_norm_left",
    "state/end/wrench_force_norm_right",
]
LEFT_VIEW = "observation.images.head_left_fisheye"
RIGHT_VIEW = "observation.images.head_right_fisheye"


class AgibotSameSideOodTest(unittest.TestCase):
    def test_report_uses_visibility_aligned_targets_and_retains_hard_diagnostics(self):
        left_values = [float(value) for value in range(8)]
        right_values = [float(value) for value in range(107, 99, -1)]
        metrics = {
            LEFT_VIEW: {
                "loss": 1.0,
                "pearson": left_values,
                "r2": left_values,
                "nrmse": left_values,
                "ccc": left_values,
            },
            RIGHT_VIEW: {
                "loss": 2.0,
                "pearson": right_values,
                "r2": right_values,
                "nrmse": right_values,
                "ccc": right_values,
            },
        }

        report = build_agibot_shot_ood_report(metrics, TARGET_FIELDS)

        self.assertEqual(report["protocol"], "agibot_shot_same_side_v1")
        self.assertEqual(
            report["primary"]["views"][LEFT_VIEW]["target_indices"],
            [0, 2, 4, 6],
        )
        self.assertEqual(
            report["primary"]["views"][RIGHT_VIEW]["target_indices"],
            [1, 3, 5, 7],
        )
        self.assertEqual(report["primary"]["macro"]["pearson"], 53.0)
        self.assertEqual(
            report["diagnostics"]["cross_side_hard"]["macro"]["pearson"],
            54.0,
        )
        self.assertEqual(
            report["diagnostics"]["full_8d_hard"]["macro"]["pearson"],
            53.5,
        )
        self.assertNotIn("loss", report["primary"]["macro"])

    def test_validation_rejects_missing_view_or_non_bilateral_target(self):
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            validate_agibot_same_side_ood(TARGET_FIELDS, [LEFT_VIEW])
        with self.assertRaisesRegex(ValueError, "_left'/'_right"):
            validate_agibot_same_side_ood(
                [*TARGET_FIELDS[:-1], "state/robot/translation_speed"],
                [LEFT_VIEW, RIGHT_VIEW],
            )


if __name__ == "__main__":
    unittest.main()
