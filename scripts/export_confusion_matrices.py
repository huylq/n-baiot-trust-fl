"""Reproduce confusion matrices and class-wise metrics for manuscript Tables 4-5."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import build_runtime_config
from src.data_loader import DataRepository
from src.federated import ExperimentRunner

IDENTITY = ["paper_table", "group", "scenario", "rho", "malicious_count", "seed", "device"]
METRICS = ["accuracy", "precision_benign", "recall_benign", "f1_benign",
           "precision_malicious", "recall_malicious", "f1_malicious", "f1_macro"]


def build_tables(records: list[dict]) -> tuple[pd.DataFrame, ...]:
    matrices, classes, evaluations = [], [], []
    seen = set()
    for record in records:
        meta = record["metadata"]
        identity = tuple(meta[name] for name in IDENTITY)
        if identity in seen:
            raise ValueError(f"Duplicate evaluation: {identity}")
        seen.add(identity)
        matrix = np.asarray(record["matrix"])
        if (matrix.shape != (2, 2) or not np.issubdtype(matrix.dtype, np.integer)
                or np.any(matrix < 0) or matrix.sum() == 0):
            raise ValueError(f"Invalid binary confusion matrix: {identity}")
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        matrices.append({**meta, "sample_count": int(matrix.sum()),
                         "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        metrics = {"accuracy": (tn + tp) / int(matrix.sum())}
        for index, label in enumerate(("benign", "malicious")):
            correct = int(matrix[index, index])
            support = int(matrix[index].sum())
            predicted = int(matrix[:, index].sum())
            precision = correct / predicted if predicted else 0.0
            recall = correct / support if support else 0.0
            f1 = 2 * correct / (support + predicted) if support + predicted else 0.0
            classes.append({**meta, "class": label, "support": support,
                            "precision": precision, "recall": recall, "f1": f1})
            metrics.update({f"precision_{label}": precision, f"recall_{label}": recall,
                            f"f1_{label}": f1})
        metrics["f1_macro"] = (metrics["f1_benign"] + metrics["f1_malicious"]) / 2
        evaluations.append({**meta, **metrics})
    matrix_frame = pd.DataFrame(matrices).sort_values(IDENTITY, na_position="first")
    class_frame = pd.DataFrame(classes).sort_values(IDENTITY + ["class"], na_position="first")
    frame = pd.DataFrame(evaluations)
    seed_keys = IDENTITY[:-1]
    for identity, items in frame.groupby(seed_keys, dropna=False):
        expected = 9 if identity[2] == "S1_local_only" else 1
        if len(items) != expected:
            raise ValueError(f"Expected {expected} device evaluations for {identity}")
    # Local-only first averages the nine device metrics within each seed.
    per_seed = frame.groupby(seed_keys, dropna=False)[METRICS].mean().reset_index()
    config_keys = seed_keys[:-1]
    summary = per_seed.groupby(config_keys, dropna=False)[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, column)) for column in summary.columns]
    counts = per_seed.groupby(config_keys, dropna=False).size().reset_index(name="seed_count")
    summary = summary.merge(counts, on=config_keys, validate="one_to_one")
    return matrix_frame, class_frame, summary, per_seed


def compare_reference(per_seed: pd.DataFrame, reference: Path, seeds: list[int]) -> None:
    with reference.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    def identity(row):
        rho = row["rho"]
        return (row["group"], row["scenario"], int(row["seed"]),
                None if rho in ("", None) or pd.isna(rho) else float(rho),
                int(row["malicious_count"]))

    expected = {identity(row): row for row in rows if int(row["seed"]) in seeds
                and (row["group"] == "G0" or (row["group"] == "G3" and row["scenario"] == "S4"))}
    actual = {identity(row): row for row in per_seed.to_dict(orient="records")}
    if set(actual) != set(expected):
        raise ValueError("Recomputed configurations do not match the selected reference rows.")
    metric_map = {"accuracy": "accuracy", "precision_malicious": "precision",
                  "recall_malicious": "recall", "f1_malicious": "f1_malicious",
                  "f1_benign": "f1_benign", "f1_macro": "f1_macro"}
    maximum = 0.0
    for key, row in actual.items():
        for name, reference_name in metric_map.items():
            difference = abs(float(row[name]) - float(expected[key][reference_name]))
            maximum = max(maximum, difference)
            if difference > 1e-12:
                raise ValueError(f"Reference mismatch: {key}, {name}, difference={difference}")
    print(f"[CHECK] {len(actual)} per-seed rows match the paper reference "
          f"within 1e-12; maximum absolute difference={maximum:.3g}.", flush=True)


def export_tables(records: list[dict], output: Path, seeds: list[int]) -> None:
    matrices, classes, summary, per_seed = build_tables(records)
    compare_reference(per_seed, ROOT / "reference/tables/all_seed_runs.csv", seeds)
    for name, frame in (("confusion_matrices.csv", matrices),
                        ("classwise_metrics.csv", classes), ("summary.csv", summary)):
        frame.to_csv(output / name, index=False, mode="x", lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "code/data/dataset20000")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/confusion_matrices")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Subset of paper seeds 0-4; default: all five")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(range(5)):
        parser.error("Choose unique paper seeds from 0, 1, 2, 3, 4.")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new --output-dir.")
    from prepare_dataset import verify_dataset
    verify_dataset(args.data_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = build_runtime_config()
    config["paths"]["raw_data_dir"] = str(args.data_dir.resolve())
    config["seeds"] = sorted(args.seeds)
    config["export"].update(write_models=False, write_figures=False, write_excel=False)
    repository = DataRepository.from_config(config)
    records = []
    with threadpool_limits(limits=1):
        for seed in sorted(args.seeds):
            seed_config = copy.deepcopy(config)
            seed_root = output / "runs" / f"seed{seed}"
            seed_config["paths"] = {"raw_data_dir": str(args.data_dir.resolve()),
                "audit_dir": str(seed_root / "audit"), "models_dir": str(seed_root / "models")}
            seed_root.mkdir(parents=True)
            with (seed_root / "runtime_config.json").open("x", encoding="utf-8") as stream:
                json.dump(seed_config, stream, indent=2)
            runner = ExperimentRunner(seed_config, repository, seed=seed)
            splits = repository.build_device_splits("binary", seed=seed)
            mean, std = repository.pooled_scaler(splits)
            np.savez(seed_root / "scaler.npz", mean=mean, std=std)

            def collect(result, scenario, count=0, device=""):
                records.append({"metadata": {
                    "paper_table": 5 if count else 4, "group": "G3" if count else "G0",
                    "scenario": scenario, "rho": 1.0 if count else None,
                    "malicious_count": count, "seed": seed, "device": device},
                    "matrix": np.asarray(result["confusion_matrix"], dtype=int).tolist()})

            collect(runner.run_centralized("binary", splits, mean, std, 2), "S1_centralized")
            for result in runner.run_local_only("binary", splits, mean, std, 2):
                collect(result, "S1_local_only", device=result["device"])
            clean = {**config["attack"], "enabled": False, "malicious_clients": []}
            collect(runner.run_federated_scenario("S1", "binary", splits, mean, std, 2, clean), "S1")
            collect(runner.run_federated("binary", splits, mean, std, 2, "trust",
                security_cfg=runner.scenario_security_config("S4"), attack_cfg=clean,
                scenario_name="S1_trust"), "S1_trust")
            for count in (1, 2, 3):
                attack = {**config["attack"], "enabled": True, "rho": 1.0,
                    "malicious_clients": config["experiments"]["malicious_clients_by_count"][str(count)]}
                collect(runner.run_federated_scenario("S4", "binary", splits, mean, std, 2, attack),
                        "S4", count=count)
            print(f"[SEED] {seed}: 15 confusion matrices collected.", flush=True)
    export_tables(records, output, sorted(args.seeds))
    print(f"[EXPORT] {len(records)} matrices, {2 * len(records)} class-wise rows, "
          f"7 configuration summaries: {output}", flush=True)


if __name__ == "__main__":
    main()
