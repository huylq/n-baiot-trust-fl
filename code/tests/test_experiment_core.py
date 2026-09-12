from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import pandas as pd

from config import build_runtime_config
from src.aggregators import (
    FoolsGoldState,
    clipped_mean,
    fltrust_aggregate,
    krum,
    krum_select,
    multi_krum,
    resolve_defense_params,
    trimmed_mean,
)
from src.data_loader import DataRepository, flip_labels
from src.evaluation import attribution_metrics, compute_metrics, paired_significance_tests
from src.model import SoftmaxRegressor
from src.security import (
    AuditChain,
    TrustState,
    clip_delta,
    compute_trust_score,
    coordinate_wise_median,
    reference_statistics,
    robust_norm_zscores,
    zero_trust_admission,
)


class DataAndModelTests(unittest.TestCase):
    def test_all_metadata_variants(self) -> None:
        cases = {
            "Danmini_Doorbell_1000sample_gafgyt_combo.csv": ("Danmini_Doorbell", "gafgyt_combo"),
            "Ennio_Doorbell_2000sample_gafgyt_junk.csv": ("Ennio_Doorbell", "gafgyt_junk"),
            "SimpleHome_XCS7_1002_WHT_1000sample_mirai_udpplain.csv": (
                "SimpleHome_XCS7_1002_WHT_Security_Camera", "mirai_udpplain"
            ),
            "Samsung_SNH_1011_N_Webcam_10000sample_benign.csv": (
                "Samsung_SNH_1011_N_Webcam", "benign"
            ),
        }
        for filename, expected in cases.items():
            self.assertEqual(DataRepository._parse_metadata(filename), expected)

    def test_label_flip_is_seeded_and_binary(self) -> None:
        labels = np.array([0, 1, 0, 1, 1])
        np.testing.assert_array_equal(flip_labels(labels, 1.0, 7), 1 - labels)
        np.testing.assert_array_equal(flip_labels(labels, 0.0, 7), labels)
        np.testing.assert_array_equal(flip_labels(labels, 0.4, 9), flip_labels(labels, 0.4, 9))

    def test_binary_softmax_delta_has_232_coordinates(self) -> None:
        model = SoftmaxRegressor(115, 2, seed=0)
        self.assertEqual(model.flattened_delta(model.get_state()).shape, (232,))


class SecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = build_runtime_config()["security"]

    def test_coordinate_median_and_median_of_norms_are_distinct(self) -> None:
        deltas = [np.array([1.0, 100.0]), np.array([2.0, 2.0]), np.array([100.0, 1.0])]
        reference, median_norm, norms = reference_statistics(deltas, "median")
        np.testing.assert_array_equal(reference, np.array([2.0, 2.0]))
        self.assertEqual(median_norm, float(np.median(norms)))
        self.assertNotAlmostEqual(median_norm, float(np.linalg.norm(reference)))
        np.testing.assert_array_equal(coordinate_wise_median(deltas), reference)

    def test_verifiable_score_does_not_use_loss(self) -> None:
        delta = np.array([1.0, 0.0])
        reference = np.array([1.0, 0.0])
        low_loss = compute_trust_score(delta, reference, 0.0, self.cfg, val_loss=0.0)
        high_loss = compute_trust_score(delta, reference, 0.0, self.cfg, val_loss=1e6)
        self.assertAlmostEqual(low_loss.s_k, 1.0)
        self.assertEqual(low_loss, high_loss)

    def test_negative_cosine_veto_and_persistent_strikes(self) -> None:
        component = compute_trust_score(
            np.array([-1.0, 0.0]), np.array([1.0, 0.0]), 0.0, self.cfg, val_loss=0.0
        )
        self.assertEqual(component.a_k, 0.0)
        self.assertEqual(component.s_k, 0.0)
        state = TrustState(["client"], self.cfg)
        outcomes = [
            state.update("client", 0.0, -1.0, 0.0, in_warmup=False) for _ in range(3)
        ]
        self.assertFalse(outcomes[1][2])
        self.assertTrue(outcomes[2][2])

    def test_warmup_rounds_never_strike(self) -> None:
        state = TrustState(["client"], self.cfg)
        for _ in range(10):
            _, strike, flagged, reasons = state.update(
                "client", 0.0, -1.0, 0.0, in_warmup=True
            )
            self.assertEqual(strike, 0)
            self.assertFalse(flagged)
            self.assertIn("negative_cosine", reasons)

    def test_convergence_noise_cosine_does_not_strike(self) -> None:
        state = TrustState(["client"], self.cfg)
        for _ in range(10):
            _, _, flagged, reasons = state.update(
                "client", 0.3, -0.05, 0.0, in_warmup=False
            )
            self.assertFalse(flagged)
            self.assertEqual(reasons, [])

    def test_honest_large_norm_client_does_not_strike(self) -> None:
        """A 6x-median honest norm (SimpleHome 1002 in G0) must not violate."""
        norms = np.array([0.5, 0.6, 0.8, 0.7, 2.2, 1.9, 1.6, 6.0, 1.0])
        zscores = robust_norm_zscores(norms, self.cfg)
        self.assertLess(float(zscores.max()), self.cfg["norm_z_strike"])
        state = TrustState(["big"], self.cfg)
        _, _, flagged, reasons = state.update(
            "big", 0.4, 0.5, float(zscores.max()), in_warmup=False
        )
        self.assertFalse(flagged)
        self.assertEqual(reasons, [])

    def test_extreme_amplification_fails_admission(self) -> None:
        norms = np.array([0.5, 0.6, 0.8, 0.7, 2.2, 1.9, 1.6, 6.0, 500.0])
        zscores = robust_norm_zscores(norms, self.cfg)
        admitted, reason = zero_trust_admission(1_000, float(zscores[-1]), self.cfg)
        self.assertFalse(admitted)
        self.assertEqual(reason, "delta_norm_robust_outlier_gate")
        self.assertEqual(zero_trust_admission(1_000, 0.0, self.cfg), (True, "accepted"))
        self.assertFalse(zero_trust_admission(10, 0.0, self.cfg)[0])

    def test_relative_trust_threshold_keeps_uniform_cohort(self) -> None:
        state = TrustState([f"c{i}" for i in range(9)], self.cfg)
        for device in state.trust:
            state.trust[device] = 0.4
        threshold = state.round_trust_threshold(state.trust.keys())
        self.assertAlmostEqual(threshold, 0.2)
        self.assertTrue(all(trust >= threshold for trust in state.trust.values()))
        state.trust["c0"] = 0.01
        self.assertLess(state.trust["c0"], state.round_trust_threshold(state.trust.keys()))

    def test_no_memory_ablation_has_no_strikes_or_flags(self) -> None:
        cfg = {**self.cfg, "use_memory": False}
        state = TrustState(["client"], cfg)
        for _ in range(5):
            trust, strike, flagged, reasons = state.update(
                "client", 0.2, -1.0, 10.0, in_warmup=False
            )
        self.assertEqual(trust, 0.2)
        self.assertEqual(strike, 0)
        self.assertFalse(flagged)
        self.assertIn("negative_cosine", reasons)
        self.assertIn("norm_robust_outlier", reasons)

    def test_clip_delta_bounds_large_updates(self) -> None:
        delta = np.array([3.0, 4.0])
        clipped = clip_delta(delta, 1.0, self.cfg)
        self.assertAlmostEqual(float(np.linalg.norm(clipped)), self.cfg["clip_multiplier"])
        small = np.array([0.3, 0.4])
        np.testing.assert_array_equal(clip_delta(small, 1.0, self.cfg), small)

    def test_audit_chain_flag_event_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chain = AuditChain(Path(directory) / "audit.jsonl")
            chain.append_flag_event(3, "client", np.array([1.0, 2.0]), ["negative_cosine"])
            chain.append({"event": "round_summary", "round": 3})
            self.assertTrue(chain.verify())
            self.assertIn("delta_hash", chain.records()[0])


class AggregatorTests(unittest.TestCase):
    """The published baseline aggregators must match their formulations."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.honest = [np.array([1.0, 1.0]) + rng.normal(0, 0.05, 2) for _ in range(7)]
        self.attackers = [np.array([-40.0, -40.0]), np.array([45.0, -50.0])]
        self.deltas = self.honest + self.attackers

    def test_krum_selects_inside_honest_cluster(self) -> None:
        selected = krum_select(self.deltas, f=2, m=1)[0]
        self.assertLess(selected, len(self.honest))
        np.testing.assert_array_equal(krum(self.deltas, 2), self.deltas[selected])

    def test_multi_krum_excludes_attackers(self) -> None:
        aggregate, selected = multi_krum(self.deltas, f=2, m=5)
        self.assertTrue(all(index < len(self.honest) for index in selected))
        self.assertLess(float(np.abs(aggregate - 1.0).max()), 0.2)

    def test_krum_requires_2f_plus_2_below_n(self) -> None:
        with self.assertRaises(ValueError):
            krum_select(self.deltas, f=4, m=1)

    def test_trimmed_mean_discards_extremes(self) -> None:
        aggregate = trimmed_mean(self.deltas, trim_k=2)
        self.assertLess(float(np.abs(aggregate - 1.0).max()), 0.2)
        untrimmed = trimmed_mean(self.deltas, trim_k=0)
        self.assertGreater(float(np.abs(untrimmed - 1.0).max()), 1.0)

    def test_clipped_mean_bounds_attacker_influence(self) -> None:
        aggregate = clipped_mean(self.deltas, clip_norm=float(np.sqrt(2.0)))
        self.assertLess(float(np.abs(aggregate).max()), 1.5)

    def test_fltrust_zeroes_opposite_updates_and_rescales(self) -> None:
        server = np.array([1.0, 1.0])
        aggregate, trust = fltrust_aggregate(self.deltas, server)
        self.assertEqual(float(trust[-2]), 0.0)  # opposite-direction attacker
        cosine = float(np.dot(aggregate, server) / (np.linalg.norm(aggregate) * np.linalg.norm(server)))
        self.assertGreater(cosine, 0.99)
        zero_aggregate, zero_trust = fltrust_aggregate([-server, -server], server)
        np.testing.assert_array_equal(zero_aggregate, np.zeros(2))
        self.assertEqual(float(zero_trust.sum()), 0.0)

    def test_foolsgold_downweights_sybils(self) -> None:
        state = FoolsGoldState(["sybil_a", "sybil_b", "honest"], dimension=4)
        sybil_update = np.array([1.0, 1.0, 0.0, 0.0])
        honest_update = np.array([0.0, 0.1, 1.0, -0.5])
        for _ in range(3):
            weights = state.weights(
                {
                    "sybil_a": sybil_update,
                    "sybil_b": sybil_update * 1.01,
                    "honest": honest_update,
                }
            )
        self.assertGreater(weights["honest"], 0.9)
        self.assertLess(weights["sybil_a"], 0.1)
        self.assertLess(weights["sybil_b"], 0.1)

    def test_resolve_defense_params_conventions(self) -> None:
        params = resolve_defense_params(9, 3, {})
        self.assertEqual(params["byzantine_f"], 3)
        self.assertEqual(params["multikrum_m"], 4)
        self.assertEqual(params["trim_k"], 3)
        clean = resolve_defense_params(9, 0, {})
        self.assertEqual(clean["byzantine_f"], 1)


class EvaluationTests(unittest.TestCase):
    def test_binary_metrics_expose_both_classes(self) -> None:
        metrics = compute_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), "binary")
        self.assertIn("f1_malicious", metrics)
        self.assertIn("f1_benign", metrics)
        self.assertIn("f1_macro", metrics)

    def test_attribution_per_round_and_cumulative(self) -> None:
        result = attribution_metrics(
            [["honest"], ["attacker"]],
            {"attacker"},
            all_clients={"honest", "attacker", "Ecobee_Thermostat"},
            audit_entries=[{"event": "client_flagged", "client": "attacker"}],
        )
        self.assertEqual(result["recall_cum"], 1.0)
        self.assertEqual(result["time_to_detection"]["attacker"], 2)
        self.assertEqual(result["false_positive_rate"], 0.5)
        self.assertEqual(result["audit_correctness"], 1.0)

    def test_paired_significance_tests(self) -> None:
        rows = []
        for seed in range(5):
            base = 0.99 + seed * 1e-4
            rows.append({"group": "G1", "scenario": "S4", "seed": seed, "rho": 1.0,
                         "malicious_count": 2, "accuracy": base})
            rows.append({"group": "G1", "scenario": "S2", "seed": seed, "rho": 1.0,
                         "malicious_count": 2, "accuracy": base - 0.05})
            rows.append({"group": "G1", "scenario": "S2m", "seed": seed, "rho": 1.0,
                         "malicious_count": 2, "accuracy": base})
        frame = paired_significance_tests(pd.DataFrame(rows), metric="accuracy")
        self.assertEqual(len(frame), 2)
        versus_s2 = frame[frame["scenario_b"] == "S2"].iloc[0]
        self.assertTrue(versus_s2["significant_at_0.05"])
        self.assertAlmostEqual(versus_s2["mean_diff_a_minus_b"], 0.05)
        versus_s2m = frame[frame["scenario_b"] == "S2m"].iloc[0]
        self.assertEqual(versus_s2m["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
