"""Recompute MLP tables and attribution diagnostics from saved predictions and logs."""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from export_client_attribution import read_chain
from mlp_common import (REFERENCE, ROOT, add_clean_differences, check_manifest,
                        load_protocol, require, summary_frame, write_json)


def compare_frames(actual, expected, name):
    require(list(actual.columns) == list(expected.columns), f"Column mismatch: {name}")
    actual, expected = actual.copy(), expected.copy()
    for column in actual.columns:
        if pd.api.types.is_numeric_dtype(expected[column]) or expected[column].isna().all():
            actual[column] = pd.to_numeric(actual[column])
            expected[column] = pd.to_numeric(expected[column])
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    except AssertionError as error:
        raise ValueError(f"Numerical reference mismatch: {name}\n{error}") from error


def collect_evidence(directory):
    protocol = load_protocol(directory / "protocol.json")
    security = protocol["security"]
    devices = protocol["clients"]
    rounds = protocol["rounds"]
    with (directory / "per_seed_metrics.csv").open(encoding="utf-8", newline="") as stream:
        supplied = list(csv.DictReader(stream))
    expected = {(seed, item["case"]) for seed in protocol["seeds"] for item in protocol["grid"]}
    require(len(supplied) == len(expected) and {(int(r["seed"]), r["case"]) for r in supplied} == expected,
            "Incomplete or duplicated MLP grid")
    rows, matrices, classes, counts, false_flags, missed, traces = [], [], [], [], [], [], []
    audit_count, decisions, predictions_count = 0, 0, 0
    for supplied_row in supplied:
        seed, case = int(supplied_row["seed"]), supplied_row["case"]
        item = next(g for g in protocol["grid"] if g["case"] == case)
        folder = (directory / supplied_row["run_dir"]).resolve()
        require(folder.is_relative_to(directory.resolve()), "Run directory escapes input directory")
        row = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        require(row["seed"] == seed and row["case"] == case and row["method"] == item["method"]
                and row["scenario"] == item["scenario"] and row["rho"] == item["rho"], "Run identity mismatch")
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            y, predicted = saved["y_true"], saved["y_pred"]
            require(y.shape == predicted.shape == (27000,), "Unexpected prediction shape")
            require(np.bincount(y).tolist() == [13500, 13500]
                    and set(np.unique(predicted)) <= {0, 1}, "Invalid binary predictions")
            matrix = confusion_matrix(y, predicted, labels=[0, 1])
            np.testing.assert_array_equal(matrix, saved["confusion_matrix"])
        predictions_count += len(y)
        calculated = {"accuracy": accuracy_score(y, predicted),
                      "precision": precision_score(y, predicted, zero_division=0),
                      "recall": recall_score(y, predicted, zero_division=0),
                      "f1": f1_score(y, predicted, zero_division=0),
                      "f1_malicious": f1_score(y, predicted, zero_division=0),
                      "f1_benign": f1_score(y, predicted, pos_label=0, zero_division=0),
                      "f1_macro": f1_score(y, predicted, average="macro", zero_division=0)}
        for key, value in calculated.items():
            require(abs(value - row[key]) <= 1e-12, f"Classification mismatch: {case}/{seed}/{key}")
        for key, value in zip(("tn", "fp", "fn", "tp"), matrix.ravel()):
            require(row[key] == int(value), "Confusion matrix count mismatch")
        identity = {"case": case, "method": item["method"], "seed": seed,
                    "rho": item["rho"], "malicious_count": row["malicious_count"]}
        matrices.append({**identity, **dict(zip(("tn", "fp", "fn", "tp"), map(int, matrix.ravel())))})
        for index, label in enumerate(("benign", "malicious")):
            correct = int(matrix[index, index])
            support, assigned = int(matrix[index].sum()), int(matrix[:, index].sum())
            classes.append({**identity, "class": label, "support": support,
                            "precision": correct / assigned if assigned else 0.0,
                            "recall": correct / support, "f1": 2 * correct / (support + assigned)})
        audit = list((folder / "audit").glob("*.jsonl"))
        require(len(audit) == 1, "Expected one audit chain per run")
        records = [r for r in read_chain(audit[0]) if r["event"] == "round_summary"]
        require([r["round"] for r in records] == list(range(1, rounds + 1)), "Missing or repeated audit round")
        for key, value in calculated.items():
            require(abs(records[-1]["metrics"][key] - value) <= 1e-12, "Audit/prediction mismatch")
        require(row["audit_chain_valid"] is True, "Invalid recorded audit status")
        audit_count += 1
        if item["method"] == "S4":
            truth = {devices[i] for i in protocol["attacker_indices"]} if item["rho"] is not None else set()
            require(row["malicious_count"] == len(truth), "Attacker count mismatch")
            windows = {device: deque(maxlen=security["window_W"]) for device in devices}
            ever, first, suppressed, suppression_rates = set(), {}, set(), []
            histories = {device: [] for device in devices}
            for record in records:
                r = record["round"]
                warm = r <= security["warmup_rounds"]
                require(record["in_warmup"] == warm, "Warm-up mismatch")
                clients = record["clients"]
                require([c["device"] for c in clients] == devices, "Client order mismatch")
                norms = np.asarray([c["delta_norm"] for c in clients])
                np.testing.assert_allclose(norms, record["norms"], rtol=0, atol=1e-12)
                lognorms = np.log(np.maximum(norms, security["eps"]))
                median = np.median(lognorms)
                scale = max(1.4826 * np.median(np.abs(lognorms - median)), security["log_norm_mad_floor"])
                zscores = (lognorms - median) / scale
                np.testing.assert_allclose(zscores, record["norm_zscores"], rtol=0, atol=1e-12)
                suppressed_now, flagged_now = set(), set()
                for ci, client in enumerate(clients):
                    device = client["device"]
                    require(client["is_malicious_ground_truth"] == (device in truth), "Ground truth mismatch")
                    cos_violation = client["cosine_to_reference"] < -security["cos_margin"]
                    norm_violation = bool(zscores[ci] > security["norm_z_strike"])
                    if not warm:
                        windows[device].append(cos_violation or norm_violation)
                        if client["aggregation_weight"] == 0 and device not in truth:
                            suppressed.add(device)
                            suppressed_now.add(device)
                    strike = sum(windows[device])
                    flag = strike >= security["strike_W_neg"]
                    require(strike == client["strike"] and flag == client["flagged"] == client["detected"],
                            f"Strike/flag mismatch: {case}/{seed}/{r}/{device}")
                    decisions += 1
                    if flag:
                        flagged_now.add(device)
                        ever.add(device)
                        first.setdefault(device, r)
                    trace = {**identity, "device": device, "round": r, "in_warmup": warm,
                             "is_malicious": device in truth, "cosine": client["cosine_to_reference"],
                             "norm_z": float(zscores[ci]), "cosine_violation": cos_violation,
                             "norm_violation": norm_violation, "strike": strike, "flagged": flag,
                             "aggregation_weight": client["aggregation_weight"]}
                    histories[device].append(trace)
                    traces.append(trace)
                require(flagged_now == set(record["flagged_clients"]) == set(record["detected_clients"]),
                        "Round/client flag mismatch")
                if not warm:
                    suppression_rates.append(len(suppressed_now) / (len(devices) - len(truth)))
                if r in {min(10, rounds), rounds}:
                    tp, fp = len(ever & truth), len(ever - truth)
                    counts.append({**identity, "horizon": r, "tp": tp, "fp": fp,
                                   "fn": len(truth - ever), "tn": len(set(devices) - truth - ever)})
                    if truth:
                        suffix = "10" if r == 10 else "cum"
                        if r == 10 or r == rounds:
                            calculated[f"precision_{suffix}"] = tp / len(ever) if ever else 0.0
                            calculated[f"recall_{suffix}"] = tp / len(truth)
                final_weight = sum(c["aggregation_weight"] for c in clients if c["device"] in truth)
                require(abs(final_weight - record["malicious_weight_share"]) <= 1e-12, "Malicious weight mismatch")
            times = [first[d] for d in truth & set(first)]
            calculated.update(flag_fpr=len(ever - truth) / (len(devices) - len(truth)),
                              suppression_fpr=len(suppressed) / (len(devices) - len(truth)),
                              mean_honest_suppressed_per_round=float(np.mean(suppression_rates)) if suppression_rates else 0.0)
            if times:
                calculated["mean_ttd"] = float(np.mean(times))
            else:
                require(row["mean_ttd"] is None, "Undefined TTD must be null")
            if truth:
                calculated["final_malicious_weight_share"] = final_weight
            for key, value in calculated.items():
                require(row[key] is not None and abs(value - row[key]) <= 1e-12,
                        f"Recomputed metric mismatch: {case}/{seed}/{key}")
            if not truth:
                for device in sorted(ever):
                    r = first[device]
                    window = [h for h in histories[device] if max(security["warmup_rounds"] + 1,
                              r - security["window_W"] + 1) <= h["round"] <= r]
                    false_flags.append({"seed": seed, "device": device, "first_flag_round": r,
                                        "cosine_violation_rounds": ";".join(str(h["round"]) for h in window if h["cosine_violation"]),
                                        "norm_violation_rounds": ";".join(str(h["round"]) for h in window if h["norm_violation"]),
                                        "min_window_cosine": min(h["cosine"] for h in window),
                                        "max_window_norm_z": max(h["norm_z"] for h in window)})
            for device in sorted(truth - ever):
                missed.extend({key: h[key] for key in ("seed", "device", "round", "in_warmup", "cosine",
                                                       "norm_z", "strike", "flagged", "aggregation_weight")}
                              for h in histories[device])
        row.update(calculated)
        rows.append(row)
    add_clean_differences(rows)
    require(len(rows) == protocol["planned_federated_runs"], "Run count does not match protocol")
    frame = pd.DataFrame(rows)
    supplied_frame = pd.read_csv(directory / "per_seed_metrics.csv")
    require(set(frame.columns) == set(supplied_frame.columns), "Per-seed columns differ")
    compare_frames(frame[list(supplied_frame.columns)], supplied_frame, "per_seed_metrics.csv")
    summary = summary_frame(rows)
    compare_frames(summary, pd.read_csv(directory / "summary.csv"), "summary.csv")
    tables = {"per_seed_metrics.csv": frame, "summary.csv": summary,
              "confusion_matrices.csv": pd.DataFrame(matrices), "classwise_metrics.csv": pd.DataFrame(classes),
              "attribution_counts.csv": pd.DataFrame(counts, columns=["case", "method", "seed", "rho", "malicious_count", "horizon", "tp", "fp", "fn", "tn"]),
              "clean_false_flag_diagnostics.csv": pd.DataFrame(false_flags, columns=["seed", "device", "first_flag_round", "cosine_violation_rounds", "norm_violation_rounds", "min_window_cosine", "max_window_norm_z"]),
              "missed_attacker_histories.csv": pd.DataFrame(missed, columns=["seed", "device", "round", "in_warmup", "cosine", "norm_z", "strike", "flagged", "aggregation_weight"]),
              "client_round_traces.csv": pd.DataFrame(traces, columns=["case", "method", "seed", "rho", "malicious_count", "device", "round", "in_warmup", "is_malicious", "cosine", "norm_z", "cosine_violation", "norm_violation", "strike", "flagged", "aggregation_weight"])}
    report = {"passed": True, "verified_predictions": predictions_count, "verified_audit_chains": audit_count,
              "recomputed_s4_client_round_decisions": decisions, "verified_seed_rows": len(rows),
              "verified_summary_rows": len(summary), "absolute_tolerance": 1e-12}
    return tables, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mlp_exports")
    args = parser.parse_args()
    directory, output = args.input_dir.resolve(), args.output_dir.resolve()
    require(not output.exists(), "Output directory already exists. Choose a new --output-dir.")
    if directory == REFERENCE.resolve():
        check_manifest(directory)
    tables, report = collect_evidence(directory)
    output.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False, mode="x", lineterminator="\n")
    write_json(output / "verification_report.json", report)
    print(json.dumps(report, indent=2))
    print(f"Exported MLP evidence: {output}")


if __name__ == "__main__":
    main()
