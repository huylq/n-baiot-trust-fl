"""Regression tests for public-release orchestration and result checks."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_results import compare, normalize


class ResultComparisonTests(unittest.TestCase):
    def setUp(self):
        self.row = {"group": "G1", "scenario": "S4", "mode": "trust", "seed": 0,
                    "rho": 1.0, "malicious_count": 2, "accuracy": 0.99}

    def test_identical(self):
        self.assertTrue(compare([self.row], [self.row])["passed"])

    def test_small_serialization_difference(self):
        other = {**self.row, "accuracy": self.row["accuracy"] + 1e-15}
        self.assertTrue(compare([other], [self.row])["passed"])

    def test_numeric_mismatch(self):
        self.assertFalse(compare([{**self.row, "accuracy": 0.98}], [self.row])["passed"])

    def test_missing_and_extra_keys(self):
        result = compare([{**self.row, "seed": 1}], [self.row])
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["missing_rows"]), 1)
        self.assertEqual(len(result["extra_rows"]), 1)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            compare([self.row, self.row], [self.row])

    def test_missing_metric_is_not_zero(self):
        self.assertFalse(compare([{**self.row, "precision": 0.0}], [self.row])["passed"])

    def test_new_diagnostic_reported(self):
        result = compare([{**self.row, "mean_honest_suppressed_per_round": 2.0}], [self.row])
        self.assertTrue(result["passed"])
        self.assertEqual(result["supplemental_values_without_reference"], 1)

    def test_normalization(self):
        self.assertIsNone(normalize(""))
        self.assertIsNone(normalize(float("nan")))
        self.assertEqual(normalize("0.99"), 0.99)
        self.assertIs(normalize("True"), True)

    def test_group_configs(self):
        for group in ("G0", "G1", "G2", "G3", "G4"):
            config = json.loads((ROOT / "configs" / f"{group}.json").read_text())
            self.assertEqual(config, {"schema_version": 2, "experiments": {"groups": [group]}})


if __name__ == "__main__":
    unittest.main()
