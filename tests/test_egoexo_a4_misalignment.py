import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.time_jepa.egoexo_a4_misaligned_dataset import (
    A4_EGOEXO_MISALIGNMENT_MODE,
    audit_take_derangement,
    build_group_derangement,
    build_take_derangement,
    write_pairing_manifest,
)


def _fake_rows(group_ids):
    return [
        {
            "take_uid": str(group_id),
            "window_key": f"{group_id}/window_{index:06d}",
        }
        for index, group_id in enumerate(group_ids)
    ]


class EgoExoA4MisalignmentTest(unittest.TestCase):
    def test_take_derangement_is_reproducible_bijective_and_cross_take(self):
        rows = _fake_rows(["a"] * 4 + ["b"] * 3 + ["c"] * 2 + ["d"])
        first = build_take_derangement(rows, seed=239)
        second = build_take_derangement(rows, seed=239)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(sorted(first.tolist()), list(range(len(rows))))
        self.assertTrue(all(index != source for index, source in enumerate(first)))
        self.assertTrue(
            all(
                rows[index]["take_uid"] != rows[int(source)]["take_uid"]
                for index, source in enumerate(first)
            )
        )

        audit = audit_take_derangement(rows, first, seed=239)
        self.assertEqual(audit["mode"], A4_EGOEXO_MISALIGNMENT_MODE)
        self.assertTrue(audit["permutation_ok"])
        self.assertEqual(audit["fixed_points"], 0)
        self.assertEqual(audit["same_take_pairs"], 0)
        self.assertTrue(audit["target_marginal_preserved_by_bijection"])
        self.assertTrue(audit["valid_mask_marginal_preserved_by_bijection"])

    def test_seed_changes_mapping_without_changing_constraints(self):
        groups = [str(index // 2) for index in range(20)]
        first = build_group_derangement(groups, seed=239)
        second = build_group_derangement(groups, seed=240)
        self.assertFalse(np.array_equal(first, second))
        for mapping in (first, second):
            self.assertEqual(sorted(mapping.tolist()), list(range(len(groups))))
            self.assertTrue(
                all(
                    index != source and groups[index] != groups[int(source)]
                    for index, source in enumerate(mapping)
                )
            )

    def test_impossible_group_distribution_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "more than half"):
            build_group_derangement(["a", "a", "a", "b"], seed=239)
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_group_derangement(["a"], seed=239)

    def test_manifest_records_every_pair_and_stable_fingerprint(self):
        rows = _fake_rows(["a", "a", "b", "b", "c", "c"])
        mapping = build_take_derangement(rows, seed=239)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pairing.json"
            first = write_pairing_manifest(path, rows, mapping, seed=239)
            second = write_pairing_manifest(path, rows, mapping, seed=239)
            self.assertEqual(first["pairing_sha256"], second["pairing_sha256"])
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
