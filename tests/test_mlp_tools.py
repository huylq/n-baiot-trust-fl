"""Integrity and configuration checks for MLP reproduction tools."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mlp_common import check_manifest, load_protocol, runtime_config, sha
from src.federated import ExperimentRunner
from src.mlp import MLPClassifier
from src.model import SoftmaxRegressor


class MLPReleaseTests(unittest.TestCase):
    def test_model_selection_keeps_softmax_default(self):
        config = runtime_config(load_protocol())
        runner = ExperimentRunner(config, None, seed=0)
        self.assertIsInstance(runner._new_model(115, 2, 0), MLPClassifier)
        config["model"].pop("architecture")
        self.assertIsInstance(runner._new_model(115, 2, 0), SoftmaxRegressor)
        config["model"]["architecture"] = "unknown"
        with self.assertRaises(ValueError):
            runner._new_model(115, 2, 0)

    def test_protocol_retains_security_parameters(self):
        from config import build_runtime_config
        protocol = load_protocol()
        self.assertEqual(protocol["security"], build_runtime_config()["security"])
        self.assertEqual(len(protocol["seeds"]) * len(protocol["grid"]), 40)
        self.assertEqual(protocol["parameter_count"], 115 * 32 + 32 + 32 * 2 + 2)

    def test_manifest_rejects_changed_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = folder / "data.csv"
            data.write_bytes(b"a,b\n1,2\n")
            with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["file", "bytes", "sha256"])
                writer.writeheader()
                writer.writerow({"file": data.name, "bytes": data.stat().st_size, "sha256": sha(data)})
            self.assertEqual(check_manifest(folder), 1)
            data.write_bytes(b"a,b\n1,3\n")
            with self.assertRaises(ValueError):
                check_manifest(folder)
            data.write_bytes(b"a,b\n1,2\n")
            (folder / "extra.csv").touch()
            with self.assertRaises(ValueError):
                check_manifest(folder)


if __name__ == "__main__":
    unittest.main()
