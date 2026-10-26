"""Check class orientation and local-only aggregation in confusion-matrix exports."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_confusion_matrices import build_tables


def evaluation(matrix, device="", scenario="S1", seed=0):
    return {"metadata": {"paper_table": 4, "group": "G0", "scenario": scenario,
            "rho": None, "malicious_count": 0, "seed": seed, "device": device},
            "matrix": matrix}


class ConfusionExportTests(unittest.TestCase):
    def test_class_orientation(self):
        matrices, classes, summary, _ = build_tables([evaluation([[9, 1], [2, 8]])])
        self.assertEqual(matrices.iloc[0][["tn", "fp", "fn", "tp"]].tolist(), [9, 1, 2, 8])
        benign = classes[classes["class"] == "benign"].iloc[0]
        malicious = classes[classes["class"] == "malicious"].iloc[0]
        self.assertAlmostEqual(benign["precision"], 9 / 11)
        self.assertAlmostEqual(benign["recall"], 9 / 10)
        self.assertAlmostEqual(malicious["precision"], 8 / 9)
        self.assertAlmostEqual(malicious["recall"], 8 / 10)
        self.assertAlmostEqual(summary.iloc[0]["accuracy_mean"], 0.85)

    def test_local_only_averages_device_metrics(self):
        records = [evaluation([[9, 1], [2, 8]], f"device{i}", "S1_local_only") for i in range(8)]
        records.append(evaluation([[5, 5], [0, 10]], "device8", "S1_local_only"))
        _, _, summary, per_seed = build_tables(records)
        expected_precision = (8 * (8 / 9) + 10 / 15) / 9
        pooled_precision = (8 * 8 + 10) / (8 * 9 + 15)
        self.assertAlmostEqual(per_seed.iloc[0]["precision_malicious"], expected_precision)
        self.assertNotAlmostEqual(expected_precision, pooled_precision)
        self.assertAlmostEqual(summary.iloc[0]["precision_malicious_mean"], expected_precision)

    def test_missing_local_device_is_rejected(self):
        with self.assertRaises(ValueError):
            build_tables([evaluation([[9, 1], [2, 8]], "device0", "S1_local_only")])

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "keep.txt"
            sentinel.write_text("existing result", encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/export_confusion_matrices.py"),
                "--output-dir", directory], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing result")


if __name__ == "__main__":
    unittest.main()
