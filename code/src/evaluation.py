from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_name: str) -> dict:
    if task_name == "binary":
        f1_malicious = float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
            "f1": f1_malicious,
            "f1_malicious": f1_malicious,
            "f1_benign": float(f1_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": f1_macro,
        "f1_macro": f1_macro,
    }


def attribution_metrics(
    flagged_per_round: Sequence[Iterable[str]] | dict[int, Iterable[str]],
    ground_truth_malicious: Iterable[str],
    all_clients: Iterable[str] | None = None,
    audit_entries: Sequence[dict] | None = None,
    ecobee_device: str = "Ecobee_Thermostat",
) -> dict:
    """Score per-round and cumulative attacker attribution.

    ``flagged_per_round`` may contain the union of clients flagged by the strike
    mechanism and clients rejected by another trust gate.  Ground truth is used
    only here for evaluation, never by the trust calculation.
    """
    malicious = set(ground_truth_malicious)
    if isinstance(flagged_per_round, dict):
        ordered = [(int(round_index), set(clients)) for round_index, clients in sorted(flagged_per_round.items())]
    else:
        ordered = [(index, set(clients)) for index, clients in enumerate(flagged_per_round, start=1)]

    if all_clients is None:
        clients = set(malicious)
        for _, flagged in ordered:
            clients.update(flagged)
    else:
        clients = set(all_clients)
    honest = clients - malicious

    cumulative: set[str] = set()
    time_to_detection: dict[str, int | None] = {client: None for client in sorted(malicious)}
    rows: list[dict] = []
    for round_index, flagged in ordered:
        true_positive = len(flagged & malicious)
        false_positive = len(flagged - malicious)
        precision = true_positive / len(flagged) if flagged else 0.0
        recall = true_positive / len(malicious) if malicious else 0.0
        cumulative.update(flagged)
        cumulative_tp = len(cumulative & malicious)
        cumulative_precision = cumulative_tp / len(cumulative) if cumulative else 0.0
        cumulative_recall = cumulative_tp / len(malicious) if malicious else 0.0
        for client in malicious & flagged:
            if time_to_detection[client] is None:
                time_to_detection[client] = round_index
        rows.append(
            {
                "round": round_index,
                "flagged_clients": sorted(flagged),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "precision_r": float(precision),
                "recall_r": float(recall),
                "precision_cum": float(cumulative_precision),
                "recall_cum": float(cumulative_recall),
                "false_positive_rate_r": float(false_positive / len(honest)) if honest else 0.0,
                "ecobee_flagged": ecobee_device in flagged,
            }
        )

    unique_false_positives = cumulative & honest
    audit_flag_entries = [
        entry for entry in (audit_entries or []) if entry.get("event") == "client_flagged"
    ]
    audit_correct = sum(entry.get("client") in malicious for entry in audit_flag_entries)
    detected_times = [round_index for round_index in time_to_detection.values() if round_index is not None]
    return {
        "per_round": rows,
        "precision_cum": rows[-1]["precision_cum"] if rows else 0.0,
        "recall_cum": rows[-1]["recall_cum"] if rows else 0.0,
        "time_to_detection": time_to_detection,
        "mean_time_to_detection": float(np.mean(detected_times)) if detected_times else None,
        "false_positive_rate": float(len(unique_false_positives) / len(honest)) if honest else 0.0,
        "ecobee_false_positive": ecobee_device in unique_false_positives,
        "audit_correctness": float(audit_correct / len(audit_flag_entries)) if audit_flag_entries else None,
    }


def paired_significance_tests(
    runs_frame: pd.DataFrame,
    metric: str = "accuracy",
    reference_scenario: str = "S4",
) -> pd.DataFrame:
    """Paired t-tests of the reference scenario against every other scenario.

    Pairing is by seed within each (group, rho, malicious_count) cell, which is
    valid because all scenarios in a cell share seeds, splits, and attackers.
    Returns one row per comparison with mean difference and p-value.
    """
    from scipy import stats

    required = {"group", "scenario", "seed", metric}
    if runs_frame.empty or not required.issubset(runs_frame.columns):
        return pd.DataFrame()

    rows: list[dict] = []

    def compare(group, rho, malicious_count, reference, scenario_b, other) -> None:
        shared = reference.index.intersection(other.index)
        if len(shared) < 2:
            return
        differences = (reference.loc[shared] - other.loc[shared]).to_numpy(dtype=float)
        if np.allclose(differences, 0.0):
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat, p_value = stats.ttest_rel(reference.loc[shared], other.loc[shared])
        rows.append(
            {
                "group": group,
                "rho": rho,
                "malicious_count": malicious_count,
                "metric": metric,
                "scenario_a": reference_scenario,
                "scenario_b": scenario_b,
                "n_seeds": int(len(shared)),
                "mean_diff_a_minus_b": float(np.mean(differences)),
                "std_diff": float(np.std(differences, ddof=1)) if len(shared) > 1 else 0.0,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_0.05": bool(p_value < 0.05),
            }
        )

    cells = runs_frame.groupby(["group", "rho", "malicious_count"], dropna=False)
    for (group, rho, malicious_count), cell in cells:
        scenarios = set(cell["scenario"])
        if reference_scenario not in scenarios:
            continue
        reference = cell[cell["scenario"] == reference_scenario].set_index("seed")[metric]
        for scenario in sorted(scenarios - {reference_scenario}):
            other = cell[cell["scenario"] == scenario].set_index("seed")[metric]
            compare(group, rho, malicious_count, reference, scenario, other)
        # Compare attack runs with the same group's clean S1 baseline.
        clean = runs_frame[
            (runs_frame["group"] == group) & (runs_frame["scenario"] == "S1")
        ]
        if not clean.empty and "S1" not in scenarios:
            compare(
                group, rho, malicious_count, reference, "S1_clean",
                clean.drop_duplicates("seed").set_index("seed")[metric],
            )
    return pd.DataFrame(rows)


def confusion_matrix_table(confusion: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(confusion, index=class_names, columns=class_names)


def dataset_profile_tables(master_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_device = master_df.groupby("device").size().reset_index(name="samples")
    by_label = master_df.groupby("label").size().reset_index(name="samples")
    by_device_label = master_df.groupby(["device", "label"]).size().reset_index(name="samples")
    missing = pd.DataFrame({
        "column": master_df.columns,
        "missing_values": [int(master_df[c].isna().sum()) for c in master_df.columns],
    })
    return {
        "device_distribution": by_device,
        "label_distribution": by_label,
        "device_label_distribution": by_device_label,
        "missing_values": missing,
    }


def attach_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(class_count)))
