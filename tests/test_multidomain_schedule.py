import unittest

import torch

from src.time_jepa.multidomain import (
    build_domain_epoch_schedule,
    compute_domain_step_quotas,
    normalize_domain_sampling_config,
    resolve_domain_epoch_size,
)


class MultiDomainScheduleTest(unittest.TestCase):
    def test_domain_sampling_config_and_epoch_size_resolution(self):
        cfg = normalize_domain_sampling_config(
            {
                "domain_sampling": {
                    "mode": "exact",
                    "epoch_size": "sum",
                    "shuffle": False,
                    "seed": 17,
                    "lambda_warmup_unit": "domain_step",
                    "quota_basis": "loader",
                }
            }
        )
        self.assertEqual(
            cfg,
            {
                "mode": "quota",
                "epoch_size": "sum_loaders",
                "shuffle": False,
                "seed": 17,
                "lambda_warmup_unit": "domain_step",
                "quota_basis": "loader_lengths",
            },
        )
        self.assertEqual(
            resolve_domain_epoch_size(
                {"covla": 98, "egoexo": 109},
                "min_loader",
            ),
            98,
        )
        self.assertEqual(
            resolve_domain_epoch_size(
                {"covla": 98, "egoexo": 109},
                "max_loader",
            ),
            109,
        )
        self.assertEqual(
            resolve_domain_epoch_size(
                {"covla": 98, "egoexo": 109},
                "sum_loaders",
            ),
            207,
        )

    def test_equal_weight_odd_quota_rotates_extra_step_across_epochs(self):
        epoch_zero = compute_domain_step_quotas(
            ["covla", "egoexo"],
            [0.5, 0.5],
            total_steps=109,
            epoch=0,
        )
        epoch_one = compute_domain_step_quotas(
            ["covla", "egoexo"],
            [0.5, 0.5],
            total_steps=109,
            epoch=1,
        )
        self.assertEqual(epoch_zero, {"covla": 55, "egoexo": 54})
        self.assertEqual(epoch_one, {"covla": 54, "egoexo": 55})
        self.assertEqual(
            {
                name: epoch_zero[name] + epoch_one[name]
                for name in epoch_zero
            },
            {"covla": 109, "egoexo": 109},
        )

    def test_quota_schedule_is_exact_and_deterministic(self):
        kwargs = {
            "domain_names": ["covla", "egoexo"],
            "weights": [0.5, 0.5],
            "total_steps": 218,
            "epoch": 3,
            "seed": 239,
            "shuffle": True,
            "device": torch.device("cpu"),
            "distributed": False,
            "rank": 0,
        }
        schedule_a, quotas_a = build_domain_epoch_schedule(**kwargs)
        schedule_b, quotas_b = build_domain_epoch_schedule(**kwargs)
        self.assertEqual(schedule_a, schedule_b)
        self.assertEqual(quotas_a, {"covla": 109, "egoexo": 109})
        self.assertEqual(quotas_b, {"covla": 109, "egoexo": 109})
        self.assertEqual(schedule_a.count("covla"), 109)
        self.assertEqual(schedule_a.count("egoexo"), 109)

    def test_loader_length_weights_reproduce_each_loader_exposure(self):
        quotas = compute_domain_step_quotas(
            ["covla", "egoexo"],
            [98, 109],
            total_steps=207,
            epoch=0,
        )
        self.assertEqual(quotas, {"covla": 98, "egoexo": 109})

    def test_invalid_domain_sampling_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            normalize_domain_sampling_config(
                {"domain_sampling": {"mode": "unsupported"}}
            )

if __name__ == "__main__":
    unittest.main()
