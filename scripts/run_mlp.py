"""Run the fixed 115-32-2 MLP experiment grid on the included N-BaIoT subset."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import copy
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

from mlp_common import (ROOT, add_clean_differences, load_protocol, require,
                        runtime_config, sha, summary_frame, write_csv, write_json)
from prepare_dataset import verify_dataset
from src.data_loader import DataRepository
from src.federated import ExperimentRunner


class EvidenceRunner(ExperimentRunner):
    def _evaluate_model(self, *args, **kwargs):
        result = super()._evaluate_model(*args, **kwargs)
        self.last_evaluation = result
        return result


def run_seed(seed, protocol, output, data_dir):
    with (output / "logs" / f"seed{seed}.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            config = runtime_config(protocol, data_dir)
            repository = DataRepository.from_config(config)
            splits = repository.build_device_splits("binary", seed=seed)
            mean, std = repository.pooled_scaler(splits)
            with (output / "splits_and_scalers" / f"seed{seed}.npz").open("xb") as stream:
                np.savez_compressed(stream, mean=mean, std=std,
                                    feature_columns=np.asarray(repository.feature_columns))
            rows = []
            for item in protocol["grid"]:
                folder = output / "runs" / f"seed{seed}" / item["case"]
                folder.mkdir(parents=True, exist_ok=False)
                run_config = copy.deepcopy(config)
                run_config["paths"] = {"audit_dir": str(folder / "audit"),
                                       "models_dir": str(folder / "models")}
                run_config["export"] = {"write_models": True, "write_figures": False, "write_excel": False}
                attacked = item["rho"] is not None
                attack = {**run_config["attack"], "enabled": attacked,
                          "rho": item["rho"] if attacked else 0.0,
                          "malicious_clients": protocol["attacker_indices"] if attacked else []}
                archived = copy.deepcopy(run_config)
                archived["paths"] = {"audit_dir": "audit", "models_dir": "models"}
                archived.pop("experiments", None)
                archived.update(effective_attack=attack, seed=seed)
                write_json(folder / "runtime_config.json", archived)
                runner = EvidenceRunner(run_config, repository, seed=seed)
                result = runner.run_federated(
                    "binary", splits, mean, std, 2, item["mode"], attack_cfg=attack,
                    security_cfg=run_config["security"], scenario_name=item["scenario"])
                require(result["audit_chain_valid"], "Audit chain failed verification")
                matrix = result["confusion_matrix"]
                evaluation = runner.last_evaluation
                with (folder / "predictions.npz").open("xb") as stream:
                    np.savez_compressed(stream, y_true=evaluation["truth"],
                                        y_pred=evaluation["predictions"], confusion_matrix=matrix)
                attribution = result["attribution"]
                window = next((r for r in attribution["per_round"] if r["round"] == 10), None) if attribution else None
                row = {"case": item["case"], "method": item["method"], "scenario": item["scenario"],
                       "seed": seed, "rho": item["rho"],
                       "malicious_count": len(protocol["attacker_indices"]) if attacked else 0,
                       "parameter_count": protocol["parameter_count"], **result["metrics"],
                       **dict(zip(("tn", "fp", "fn", "tp"), map(int, matrix.ravel()))),
                       "precision_10": window["precision_cum"] if window and attacked else None,
                       "recall_10": window["recall_cum"] if window and attacked else None,
                       "precision_cum": attribution["precision_cum"] if attribution and attacked else None,
                       "recall_cum": attribution["recall_cum"] if attribution and attacked else None,
                       "mean_ttd": attribution["mean_time_to_detection"] if attribution else None,
                       "flag_fpr": attribution["false_positive_rate"] if attribution else None,
                       "suppression_fpr": attribution["suppression_fpr"] if attribution else None,
                       "mean_honest_suppressed_per_round": attribution["mean_honest_suppressed_per_round"] if attribution else None,
                       "final_malicious_weight_share": attribution["final_malicious_weight_share"] if attribution and attacked else None,
                       "audit_chain_valid": result["audit_chain_valid"],
                       "run_dir": folder.relative_to(output).as_posix()}
                write_json(folder / "result.json", row)
                rows.append(row)
            return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mlp")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "code/data/dataset20000")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--cases", nargs="+", help="Subset of case names for a smoke test")
    args = parser.parse_args()
    protocol = load_protocol()
    require(1 <= args.rounds <= 50 and args.workers >= 1, "Invalid rounds or worker count")
    if args.seeds is not None:
        require(len(set(args.seeds)) == len(args.seeds) and set(args.seeds) <= set(protocol["seeds"]), "Invalid seeds")
        protocol["seeds"] = args.seeds
    if args.cases is not None:
        require(set(args.cases) <= {g["case"] for g in protocol["grid"]}, "Unknown case")
        protocol["grid"] = [g for g in protocol["grid"] if g["case"] in args.cases]
    protocol["rounds"] = protocol["training"]["federated_rounds"] = args.rounds
    protocol["planned_federated_runs"] = len(protocol["seeds"]) * len(protocol["grid"])
    output = args.output_dir.resolve()
    require(not output.exists(), "Output directory already exists. Choose a new --output-dir.")
    verify_dataset(args.data_dir.resolve())
    require(sha(ROOT / "data/dataset20000_manifest.csv") == protocol["dataset_manifest_sha256"], "Dataset manifest changed")
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    (output / "splits_and_scalers").mkdir()
    write_json(output / "protocol.json", protocol)
    rows = []
    if args.workers == 1:
        for seed in protocol["seeds"]:
            rows.extend(run_seed(seed, protocol, output, args.data_dir.resolve()))
            print(f"Completed seed {seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(run_seed, seed, protocol, output, args.data_dir.resolve()): seed
                       for seed in protocol["seeds"]}
            for future in as_completed(pending):
                rows.extend(future.result())
                print(f"Completed seed {pending[future]}", flush=True)
    order = {item["case"]: i for i, item in enumerate(protocol["grid"])}
    rows.sort(key=lambda row: (row["seed"], order[row["case"]]))
    add_clean_differences(rows)
    write_csv(output / "per_seed_metrics.csv", rows)
    summary_frame(rows).to_csv(output / "summary.csv", index=False, mode="x", lineterminator="\n")
    print(f"Completed {len(rows)} runs: {output}")


if __name__ == "__main__":
    main()
