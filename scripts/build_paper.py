"""Route the original table and figure builders to a completed result index."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
GROUPS = ["G0", "G1", "G2", "G3", "G4"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "outputs" / "runs.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    runs = {}
    for group in GROUPS:
        record = index.get(group, {})
        if record.get("status") != "complete" or record.get("seeds") != [0, 1, 2, 3, 4] or record.get("rounds") != 50:
            parser.error(f"{group} must be complete with seeds 0-4 and 50 rounds.")
        runs[group] = (args.index.resolve().parent / record["run_root"]).resolve()
        config = json.loads((runs[group] / "runtime_config.json").read_text(encoding="utf-8"))
        if config["seeds"] != [0, 1, 2, 3, 4] or config["model"]["federated_rounds"] != 50:
            parser.error(f"{group} runtime configuration disagrees with the index.")
    output = args.output_dir.resolve()
    if output.name != "paper":
        parser.error("--output-dir must end in 'paper' (the original figure builder's table layout).")
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory; reference artifacts will not be overwritten.")
    package = load_module("paper_package", ROOT / "code" / "build_paper_package.py")
    package.RUNS_DIR = args.index.resolve().parent
    package.OFFICIAL_RUNS = {group: str(path) for group, path in runs.items()}
    package.PAPER = output
    package.main()
    figures = load_module("paper_figures", ROOT / "code" / "build_paper_figures.py")
    figures.PROJECT = output.parent
    figures.OUT = output / "figures"
    for group, path in runs.items():
        setattr(figures, f"{group}_RUN", path)
    read_rounds = figures.read_rounds
    legacy = runs["G1"] / "audit" / "audit_S4_binary_seed0.jsonl"
    current = runs["G1"] / "audit" / "audit_S4_binary_rho1_m2_seed0.jsonl"

    def read_current_audit(path):
        return read_rounds(current if path == legacy and not legacy.exists() else path)

    figures.read_rounds = read_current_audit
    for name in ("fig1_rho_boundary", "fig2_main_comparison", "fig3_trust_trajectories",
                 "fig4_attribution_rounds", "fig5_g0_diagnostics", "fig6_ablation", "fig7_malicious_count"):
        getattr(figures, name)()
    print("WARNING: inherited figure annotations have known limitations; see docs/reproducibility.md.")
    print(f"[PAPER] {output}")


if __name__ == "__main__":
    main()
