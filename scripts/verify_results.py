"""Compare per-seed metrics with the supplied numeric reference."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("group", "scenario", "mode", "seed", "rho", "malicious_count")
METRICS = ("accuracy", "precision", "recall", "f1", "f1_malicious", "f1_benign", "f1_macro",
           "attribution_precision", "attribution_recall", "mean_time_to_detection",
           "false_positive_rate", "suppression_fpr", "final_malicious_weight_share",
           "ecobee_false_positive", "audit_correctness", "audit_chain_valid",
           "mean_honest_suppressed_per_round")


def normalize(value):
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        number = float(value)
        return None if math.isnan(number) else number
    except (ValueError, TypeError):
        return value


def key(row):
    return tuple(normalize(row.get(column)) for column in KEYS)


def compare(actual, expected, tolerance=1e-12):
    observed = {key(row): row for row in actual}
    reference = {key(row): row for row in expected}
    if len(observed) != len(actual) or len(reference) != len(expected):
        raise ValueError("Duplicate experiment keys")
    differences = []
    unchecked = 0
    compared = 0
    max_difference = 0.0
    for identity in reference.keys() & observed.keys():
        for column in METRICS:
            left = normalize(observed[identity].get(column))
            right = normalize(reference[identity].get(column))
            # Earlier reference exports lack this supplemental diagnostic.
            if column == "mean_honest_suppressed_per_round" and right is None and left is not None:
                unchecked += 1
                continue
            compared += 1
            if isinstance(left, (float, int)) and isinstance(right, (float, int)):
                difference = abs(left - right)
                max_difference = max(max_difference, difference)
                equal = difference <= tolerance
            else:
                equal = left == right
            if not equal:
                differences.append({"key": identity, "column": column, "actual": left, "expected": right})
    missing = sorted(reference.keys() - observed.keys(), key=str)
    extra = sorted(observed.keys() - reference.keys(), key=str)
    return {"passed": not differences and not missing and not extra,
            "actual_rows": len(actual), "expected_rows": len(expected),
            "compared_values": compared, "max_absolute_difference": max_difference,
            "supplemental_values_without_reference": unchecked,
            "missing_rows": missing, "extra_rows": extra, "differences": differences}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "outputs" / "runs.json")
    parser.add_argument("--allow-partial", action="store_true", help="Check only the groups/seeds named in the index")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    groups = {"G0", "G1", "G2", "G3", "G4"}
    if not index or not set(index).issubset(groups):
        parser.error("Index must contain known experiment groups.")
    if not args.allow_partial and (set(index) != groups or any(r.get("seeds") != [0, 1, 2, 3, 4] for r in index.values())):
        parser.error("Full verification requires all five groups and seeds; use --allow-partial for a subset.")
    with (ROOT / "reference" / "tables" / "all_seed_runs.csv").open(encoding="utf-8-sig") as handle:
        reference = list(csv.DictReader(handle))
    actual, expected = [], []
    for group, record in index.items():
        if record.get("status") != "complete" or record.get("rounds") != 50:
            parser.error(f"{group} must be complete with 50 rounds; smoke runs are not paper results.")
        run = (args.index.resolve().parent / record["run_root"]).resolve()
        config = json.loads((run / "runtime_config.json").read_text(encoding="utf-8"))
        if config["seeds"] != record["seeds"] or config["model"]["federated_rounds"] != 50:
            parser.error(f"{group} runtime configuration disagrees with its index.")
        incremental = run / "logs" / "rows_incremental.jsonl"
        if incremental.exists():
            rows = [json.loads(line) for line in incremental.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            import pandas as pd
            rows = pd.read_excel(run / "tables" / "experiment_results.xlsx", sheet_name="all_seed_runs").to_dict("records")
        actual.extend(rows)
        expected.extend(row for row in reference if row["official_group"] == group and int(row["seed"]) in record["seeds"])
    result = compare(actual, expected)
    result["scope"] = "selected groups/seeds" if args.allow_partial else "full G0-G4, five seeds"
    payload = json.dumps(result, indent=2, allow_nan=False)
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
