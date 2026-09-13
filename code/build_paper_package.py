"""Export CSV tables, Excel workbooks and an input-run index from experiment results."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.evaluation import paired_significance_tests

PROJECT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT / "outputs" / "runs"
PAPER = PROJECT / "paper"

OFFICIAL_RUNS = {
    "G0": "run_20260718_082416",
    "G1": "run_20260718_103707",
    "G2": "run_20260718_144214",
    "G3": "run_20260718_153631",
    "G4": "run_20260718_113441",
}

SCENARIO_LABELS = {
    "S1": "FedAvg (clean, no attack)",
    "S1_centralized": "Centralized (clean)",
    "S1_local_only": "Local-only (clean)",
    "S1_trust": "Trust v2 (clean)",
    "S2": "FedAvg (no defense)",
    "S2m": "Coordinate-wise median",
    "S3": "Trust ablation S3 (mean ref + self-loss, no memory)",
    "S4": "Trust v2 (proposed)",
    "S4_minus_CT1": "S4 w/o CT1 (mean reference)",
    "S4_minus_verif": "S4 + self-reported loss",
    "S4_minus_CT3": "S4 w/o CT3 (no memory)",
    "B_krum": "Krum (Blanchard et al., 2017)",
    "B_mkrum": "Multi-Krum (Blanchard et al., 2017)",
    "B_tmean": "Trimmed mean (Yin et al., 2018)",
    "B_cmean": "Norm-clipped mean",
    "B_fltrust": "FLTrust (Cao et al., 2021)",
    "B_fgold": "FoolsGold (Fung et al., 2020)",
}


def load_rows(group: str, run_id: str) -> pd.DataFrame:
    run_root = RUNS_DIR / run_id
    incremental = run_root / "logs" / "rows_incremental.jsonl"
    if incremental.exists():
        rows = [json.loads(line) for line in incremental.read_text(encoding="utf-8").splitlines()]
        frame = pd.DataFrame(rows)
    else:  # Workbook fallback for runs without incremental logs.
        frame = pd.read_excel(run_root / "tables" / "experiment_results.xlsx", sheet_name="all_seed_runs")
    frame["run_id"] = run_id
    frame["official_group"] = group
    return frame


def summary_frame(run_id: str) -> tuple[pd.DataFrame, list[dict]]:
    payload = json.loads((RUNS_DIR / run_id / "logs" / "run_summary.json").read_text(encoding="utf-8"))
    return pd.DataFrame(payload["seeds_summary"]), payload.get("rho_sweep", [])


def formatted_main_table(summary: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = summary[summary["group"] == group].copy()
    rows["method"] = rows["scenario"].map(lambda s: SCENARIO_LABELS.get(s, s))
    out = pd.DataFrame({
        "scenario": rows["scenario"],
        "method": rows["method"],
        "rho": rows.get("rho"),
        "malicious_count": rows.get("malicious_count"),
        "accuracy_pct_mean": (rows["accuracy_mean"] * 100).round(3),
        "accuracy_pct_std": (rows["accuracy_std"] * 100).round(3),
        "f1_macro_pct_mean": (rows["f1_macro_mean"] * 100).round(3),
        "f1_macro_pct_std": (rows["f1_macro_std"] * 100).round(3),
    })
    for source, target in (
        ("attribution_precision_mean", "attribution_precision"),
        ("attribution_recall_mean", "attribution_recall"),
        ("mean_time_to_detection_mean", "time_to_detection_rounds"),
        ("false_positive_rate_mean", "flag_false_positive_rate"),
        ("suppression_fpr_mean", "suppression_fpr"),
        ("mean_honest_suppressed_per_round_mean", "mean_honest_suppressed_per_round"),
        ("final_malicious_weight_share_mean", "final_malicious_weight_share"),
    ):
        if source in rows.columns:
            out[target] = pd.to_numeric(rows[source], errors="coerce").round(4)
    return out.sort_values("accuracy_pct_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    tables = PAPER / "tables"
    figures = PAPER / "figures"
    excel = PAPER / "excel"
    for directory in (tables, figures, excel):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Consolidated per-seed rows (the primary CSV for any recomputation).
    all_rows = pd.concat(
        [load_rows(group, run_id) for group, run_id in OFFICIAL_RUNS.items()],
        ignore_index=True, sort=False,
    )
    all_rows.to_csv(tables / "all_seed_runs.csv", index=False)

    # 2. Per-group summary CSVs + formatted paper tables.
    summaries = {}
    for group, run_id in OFFICIAL_RUNS.items():
        summary, rho_sweep = summary_frame(run_id)
        summaries[group] = summary
        summary.to_csv(tables / f"{group.lower()}_seeds_summary.csv", index=False)
        if rho_sweep:
            pd.DataFrame(rho_sweep).to_csv(tables / f"{group.lower()}_rho_sweep.csv", index=False)

    formatted_main_table(summaries["G0"], "G0").to_csv(tables / "table1_g0_clean_baselines.csv", index=False)
    formatted_main_table(summaries["G1"], "G1").to_csv(tables / "table2_g1_main_comparison.csv", index=False)
    formatted_main_table(summaries["G3"], "G3").to_csv(tables / "table4_g3_malicious_count.csv", index=False)
    formatted_main_table(summaries["G4"], "G4").to_csv(tables / "table5_g4_ablation.csv", index=False)
    g2 = pd.read_csv(tables / "g2_rho_sweep.csv")
    g2_table = pd.DataFrame({
        "rho": g2["rho"],
        "s2_accuracy_drop_pct": (g2["s2_accuracy_drop"] * 100).round(3),
        "s2_accuracy_drop_pct_std": (g2["s2_accuracy_drop_std"] * 100).round(3),
        "s4_accuracy_drop_pct": (g2["s4_accuracy_drop"] * 100).round(3),
        "s4_accuracy_drop_pct_std": (g2["s4_accuracy_drop_std"] * 100).round(3),
        "fltrust_accuracy_drop_pct": (g2["fltrust_accuracy_drop"] * 100).round(3),
        "fltrust_accuracy_drop_pct_std": (g2["fltrust_accuracy_drop_std"] * 100).round(3),
        "s4_attribution_recall": g2["recall"].round(3),
        "s4_attribution_precision": g2["precision"].round(3),
    })
    g2_table.to_csv(tables / "table3_g2_rho_boundary.csv", index=False)

    # 3. Fresh significance tests over the consolidated rows (includes the
    #    cross-cell S4-vs-S1_clean comparison for every group).
    significance = pd.concat(
        [
            frame for frame in (
                paired_significance_tests(all_rows[all_rows["official_group"] == group], metric=metric)
                for group in OFFICIAL_RUNS
                for metric in ("accuracy", "f1_macro")
            ) if not frame.empty
        ],
        ignore_index=True,
    )
    significance.to_csv(tables / "table6_significance_paired_ttests.csv", index=False)

    # Detection-window attribution: cumulative strike flags in rounds 1-10.
    early_rows: list[dict] = []
    seen_cells: set[tuple] = set()
    for group, run_id in OFFICIAL_RUNS.items():
        audit_dir = RUNS_DIR / run_id / "audit"
        if not audit_dir.exists():
            continue
        for audit_path in sorted(audit_dir.glob("audit_S4*.jsonl")):
            flagged: set[str] = set()
            malicious: set[str] = set()
            scenario = rho = count = seed = None
            with audit_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("event") != "round_summary":
                        continue
                    scenario = record["scenario"]
                    seed = record["seed"]
                    rho = record.get("rho")
                    for client in record["clients"]:
                        if client["is_malicious_ground_truth"]:
                            malicious.add(client["device"])
                    if record["round"] <= 10:
                        flagged.update(record["flagged_clients"])
            if not malicious or scenario is None:
                continue
            count = len(malicious)
            cell = (scenario, rho, count, seed)
            if cell in seen_cells:  # identical config reruns across groups
                continue
            seen_cells.add(cell)
            true_positive = len(flagged & malicious)
            early_rows.append({
                "group": group, "scenario": scenario, "rho": rho,
                "malicious_count": count, "seed": seed,
                "flagged_in_first_10_rounds": len(flagged),
                "precision_at_detection": true_positive / len(flagged) if flagged else None,
                "recall_at_detection": true_positive / len(malicious),
            })
    if early_rows:
        early = pd.DataFrame(early_rows)
        early.to_csv(tables / "table7_detection_window_attribution.csv", index=False)
        early_summary = (
            early.groupby(["scenario", "rho", "malicious_count"], dropna=False)
            [["precision_at_detection", "recall_at_detection"]]
            .mean().reset_index()
        )
        early_summary.to_csv(tables / "table7_detection_window_summary.csv", index=False)

    # 4. Workbooks. Publication figures are produced by build_paper_figures.py
    #    (run it after this script); raw diagnostic PNGs stay in the run dirs.
    for group, run_id in OFFICIAL_RUNS.items():
        workbook = RUNS_DIR / run_id / "tables" / "experiment_results.xlsx"
        if workbook.exists():
            shutil.copy2(workbook, excel / f"{group.lower()}_experiment_results.xlsx")

    # Provenance index for the generated tables and figures.
    lines = [
        "# Experiment tables and figures",
        "",
        "Generated from the run artifacts listed below.",
        "Tables summarize per-seed metrics from completed experiments.",
        "Figure scripts write PDF and PNG versions.",
        "See the repository's docs/reproducibility.md for known limitations.",
        "",
        "## Input runs",
        "",
    ]
    lines += [f"- {group}: {run_id}" for group, run_id in OFFICIAL_RUNS.items()]
    (PAPER / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[paper] tables: {len(list(tables.glob('*.csv')))} CSV")
    print(f"[paper] excel: {len(list(excel.glob('*.xlsx')))} workbooks")
    print(f"[paper] README: {PAPER / 'README.md'}")


if __name__ == "__main__":
    main()
