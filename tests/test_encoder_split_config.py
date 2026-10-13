import tempfile
import unittest
from pathlib import Path

from covla_jepa_rjob.split_config import (
    configured_encoder_domains,
    configured_encoder_train_split,
    encoder_train_split_required,
    expected_encoder_train_videos,
    resolve_encoder_split_enabled,
)


class EncoderSplitConfigTest(unittest.TestCase):
    def setUp(self):
        self.params = {
            "data": {
                "dataset_type": "CoVLAVideoDataset",
                "encoder_train_split": "time-jepa/splits/covla_weather_ood/covla_encoder_train.json",
                "require_encoder_train_split": True,
                "expected_encoder_train_videos": 6305,
            },
            "time_jepa": {"enabled": True},
        }

    def test_single_domain_covla_is_detected_in_auto_mode(self):
        self.assertEqual(configured_encoder_domains(self.params), {"covla"})
        self.assertTrue(resolve_encoder_split_enabled("auto", self.params, "covla"))

    def test_relative_single_domain_split_is_resolved_from_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expected = (
                Path(tmp_dir)
                / "time-jepa/splits/covla_weather_ood/covla_encoder_train.json"
            ).resolve()
            actual = configured_encoder_train_split(
                self.params,
                "covla",
                repo_root=tmp_dir,
            )
        self.assertEqual(actual, str(expected))

    def test_required_split_cannot_be_disabled(self):
        self.assertTrue(encoder_train_split_required(self.params, "covla"))
        with self.assertRaisesRegex(RuntimeError, "requires an encoder-train split"):
            resolve_encoder_split_enabled("0", self.params, "covla")

    def test_expected_video_count_is_read_from_single_domain_data(self):
        self.assertEqual(expected_encoder_train_videos(self.params, "covla"), 6305)

    def test_conflicting_top_level_and_domain_splits_are_rejected(self):
        self.params["time_jepa"] = {
            "mode": "multi_domain",
            "domains": {
                "covla": {
                    "domain": "covla",
                    "encoder_train_split": "different.json",
                }
            },
        }
        with self.assertRaisesRegex(RuntimeError, "Conflicting encoder_train_split"):
            configured_encoder_train_split(self.params, "covla")


if __name__ == "__main__":
    unittest.main()
