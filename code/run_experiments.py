from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path

import pandas as pd

from config import SCHEMA_VERSION, build_runtime_config
from src.data_loader import DataRepository
from src.evaluation import paired_significance_tests
from src.excel_export import export_experiment_workbook
from src.federated import ExperimentRunner
from src.plotting import (
    plot_attribution_curves,
    plot_class_distribution,
    plot_rho_boundary,
    plot_robustness_bars,
    plot_s1_diagnostics,
)
from src.utils import ensure_directories, set_random_seed, timestamp_string


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: Path | None) -> dict:
    config = build_runtime_config()
    if config_path is None:
        return config
    override = json.loads(config_path.read_text(encoding="utf-8"))
    if override.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"Expected configuration schema_version={SCHEMA_VERSION}")
    merged = _deep_merge(config, override)
    # Paths are relocatable; project_root always follows this checkout unless explicitly overridden.
    return merged


def prepare_run_paths(config: dict) -> dict:
    run_id = f"run_{timestamp_string()}"
    run_root = Path(config["paths"]["runs_root"]) / run_id
    suffix = 1
    while run_root.exists():
        run_root = Path(config["paths"]["runs_root"]) / f"{run_id}_{suffix}"
        suffix += 1
    config["run_id"] = run_root.name
    config["paths"].update(
        {
            "run_root": str(run_root),
            "tables_dir": str(run_root / "tables"),
            "figures_dir": str(run_root / "figures"),
            "audit_dir": str(run_root / "audit"),
            "models_dir": str(run_root / "models"),
            "logs_dir": str(run_root / "logs"),
        }
    )
    return config


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nine-device N-BaIoT poisoning experiment grid")
    parser.add_argument(
        "--groups", nargs="+", choices=["G0", "G1", "G2", "G3", "G4"],
        help="Experiment groups to run. Defaults to the config (G0 only).",
    )
    parser.add_argument("--config", type=Path, help="Optional schema-v2 JSON override")
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional seed override")
    parser.add_argument("--rounds", type=int, help="Optional round override for a smoke test")
    parser.add_argument("--no-models", action="store_true", help="Skip .npz model files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seeds:
        config["seeds"] = args.seeds
    if args.rounds is not None:
        if args.rounds <= 0:
            raise ValueError("--rounds must be positive")
        config["model"]["federated_rounds"] = args.rounds
    if args.no_models:
        config["export"]["write_models"] = False
    groups = args.groups or config["experiments"]["groups"]
    config["experiments"]["groups"] = groups
    config = prepare_run_paths(config)

    set_random_seed(config["random_seed"])
    ensure_directories(config)
    run_root = Path(config["paths"]["run_root"])
    config_path = run_root / "runtime_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    repo = DataRepository.from_config(config)
    print(f"[DATA] loaded {len(repo.master_df):,} rows, {len(repo.devices)} devices, {len(repo.feature_columns)} features")
    print(repo.device_profile().to_string(index=False))

    runner = ExperimentRunner(config=config, repo=repo)
    grid_output = runner.run_experiment_grid(groups)

    figures_dir = Path(config["paths"]["figures_dir"])
    write_figures = bool(config["export"].get("write_figures", True))
    if write_figures:
        plot_class_distribution(repo.master_df, figures_dir / "class_distribution.png")

    if write_figures and "G0" in groups:
        first_seed = int(config["seeds"][0])
        diagnostic = grid_output["detailed_runs"].get(f"G0_S1_trust_seed{first_seed}")
        if diagnostic:
            plot_s1_diagnostics(
                diagnostic["rounds"], figures_dir,
                warmup_rounds=int(config["security"].get("warmup_rounds", 5)),
            )
            plot_attribution_curves(
                diagnostic["attribution"], figures_dir / "s1_attribution_curves.png",
                "G0/S1 clean attribution (false-positive diagnostic)",
            )

    if write_figures and not grid_output["rho_sweep"].empty:
        plot_rho_boundary(grid_output["rho_sweep"], figures_dir / "rho_boundary.png")

    if write_figures:
        for group in ("G1", "G3", "G4"):
            summary = grid_output["seeds_summary"]
            group_summary = summary[summary["group"] == group]
            if not group_summary.empty:
                plot_robustness_bars(group_summary, figures_dir / f"robustness_{group.lower()}.png")

        plotted_scenarios: set[str] = set()
        for result in grid_output["detailed_runs"].values():
            scenario = result["scenario"]
            if not result.get("attribution") or scenario in plotted_scenarios or scenario == "S1_trust":
                continue
            plotted_scenarios.add(scenario)
            safe_scenario = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario)
            plot_attribution_curves(
                result["attribution"],
                figures_dir / f"attribution_{safe_scenario}.png",
                f"Attribution precision/recall: {scenario}",
            )

    significance_frames = [
        paired_significance_tests(grid_output["runs"], metric=metric)
        for metric in ("accuracy", "f1_macro")
    ]
    significance_frames = [frame for frame in significance_frames if not frame.empty]
    grid_output["significance"] = (
        pd.concat(significance_frames, ignore_index=True) if significance_frames else pd.DataFrame()
    )

    workbook_path = None
    if config["export"].get("write_excel", True):
        workbook_path = export_experiment_workbook(config, grid_output)
    audit_column = grid_output["runs"].get("audit_chain_valid")
    audit_chains_valid = True if audit_column is None else bool(audit_column.fillna(True).all())
    summary_payload = _json_safe({
        "run_id": config["run_id"],
        "groups": groups,
        "seeds": config["seeds"],
        "device_count": len(repo.devices),
        "sample_count": len(repo.master_df),
        "feature_count": len(repo.feature_columns),
        "delta_dimension_binary": len(repo.feature_columns) * 2 + 2,
        "seeds_summary": grid_output["seeds_summary"].to_dict(orient="records"),
        "rho_sweep": grid_output["rho_sweep"].to_dict(orient="records"),
        "audit_chains_valid": audit_chains_valid,
    })
    summary_path = Path(config["paths"]["logs_dir"]) / "run_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"[DONE] results: {run_root}")
    print(f"[DONE] workbook: {workbook_path}")
    if groups == ["G0"]:
        print("[STOP] G0 is complete. Review the three S1 diagnostics before running G1-G4.")


if __name__ == "__main__":
    main()
