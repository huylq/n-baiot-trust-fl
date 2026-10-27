"""Check strike-window timing, cumulative counts, and audit integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_client_attribution import read_chain, replay_rounds, summarize_run

DEVICES = [f"client{i}" for i in range(9)]
SECURITY = {"window_W": 5, "warmup_rounds": 1, "eps": 1e-6,
            "log_norm_mad_floor": 0.05, "cos_margin": 0.1,
            "norm_z_strike": 3.5, "strike_W_neg": 3}


def round_record(number, cosine, strike, flag, large_norm=False):
    norms = [float(np.exp(0.5)) if large_norm else 1.0] + [1.0] * 8
    clients = [{"device": device, "delta_norm": norms[i],
                "cosine_to_reference": cosine if i == 0 else 0.5,
                "is_malicious_ground_truth": i == 0,
                "strike": strike if i == 0 else 0,
                "flagged": flag if i == 0 else False,
                "detected": flag if i == 0 else False}
               for i, device in enumerate(DEVICES)]
    flagged = [DEVICES[0]] if flag else []
    z = float(np.log(norms[0]) / 0.05)
    return {"round": number, "in_warmup": number == 1, "clients": clients,
            "norms": norms, "norm_zscores": [z] + [0.0] * 8,
            "reference_norm": 1.0, "flagged_clients": flagged,
            "detected_clients": flagged}


class AttributionExportTests(unittest.TestCase):
    def test_nonconsecutive_strikes_warmup_and_flag_removal(self):
        cosines = [-0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5]
        strikes = [0, 1, 2, 2, 3, 3, 2, 1]
        rounds = [round_record(i + 1, cosines[i], strikes[i], i in (4, 5)) for i in range(8)]
        traces = replay_rounds(rounds, DEVICES, {DEVICES[0]}, SECURITY, 1, 0)
        self.assertEqual([r["round"] for r in traces if r["flagged"]], [5, 6])
        summary, events = summarize_run(traces, DEVICES, {DEVICES[0]}, 1, 0, 5)
        self.assertEqual(summary["TP_50"], 1)
        self.assertEqual(summary["FP_50"], 0)
        self.assertEqual(summary["mean_ttd"], 5)
        self.assertEqual(events, [])

    def test_norm_outliers_can_trigger_without_negative_cosine(self):
        rounds = [round_record(1, 0.5, 0, False)]
        rounds += [round_record(i, 0.5, i - 1, i == 4, large_norm=True) for i in (2, 3, 4)]
        traces = replay_rounds(rounds, DEVICES, {DEVICES[0]}, SECURITY, 1, 0)
        flags = [r for r in traces if r["flagged"]]
        self.assertEqual(len(flags), 1)
        self.assertTrue(flags[0]["norm_violation"])
        self.assertFalse(flags[0]["cosine_violation"])

    def test_corrupted_audit_record_is_rejected(self):
        payload = {"previous_hash": "GENESIS", "event": "example", "value": 1}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        record = {**payload, "current_hash": digest}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(len(read_chain(path)), 1)
            record["value"] = 2
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid hash"):
                read_chain(path)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "keep.txt"
            sentinel.write_text("existing result", encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/export_client_attribution.py"),
                                     "--output-dir", directory], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing result")


if __name__ == "__main__":
    unittest.main()
