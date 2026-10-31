"""Verify MLP reference files, test predictions, checkpoints and attribution decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from export_mlp_results import collect_evidence, compare_frames
from mlp_common import (REFERENCE, ROOT, check_manifest, dataset_manifest_digest,
                        load_protocol, require, runtime_config, sha)
from prepare_dataset import verify_dataset
from src.data_loader import DataRepository


def verify(directory, data_dir, compare_reference=False):
    manifest_files = check_manifest(directory) if directory == REFERENCE.resolve() else None
    protocol = load_protocol(directory / "protocol.json")
    require(dataset_manifest_digest() == protocol["dataset_manifest_sha256"], "Dataset manifest changed")
    verify_dataset(data_dir)
    tables, report = collect_evidence(directory)
    if compare_reference:
        for name in ("per_seed_metrics.csv", "summary.csv"):
            compare_frames(tables[name], pd.read_csv(REFERENCE / name), name)
        report["matched_table6_reference"] = True
    if manifest_files is not None:
        for _, entry in pd.read_csv(directory / "source_manifest.csv").iterrows():
            path = (ROOT / entry["file"]).resolve()
            require(path.is_relative_to(ROOT.resolve()) and sha(path) == entry["sha256"],
                    f"Public source checksum mismatch: {entry['file']}")
    for name, frame in tables.items():
        if name == "client_round_traces.csv":
            continue
        path = directory / name
        if path.exists():
            compare_frames(frame, pd.read_csv(path), name)
    repository = DataRepository.from_config(runtime_config(protocol, data_dir))
    models = 0
    for seed in protocol["seeds"]:
        splits = repository.build_device_splits("binary", seed=seed)
        mean, std = repository.pooled_scaler(splits)
        with np.load(directory / f"splits_and_scalers/seed{seed}.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(mean, saved["mean"])
            np.testing.assert_array_equal(std, saved["std"])
            np.testing.assert_array_equal(repository.feature_columns, saved["feature_columns"])
        x = np.concatenate([repository.transform_features(splits[d].test, mean, std) for d in repository.devices])
        y = np.concatenate([repository.encode_labels(splits[d].test, "binary") for d in repository.devices])
        for _, row in tables["per_seed_metrics.csv"].query("seed == @seed").iterrows():
            folder = directory / row["run_dir"]
            checkpoints = list((folder / "models").glob("*.npz"))
            require(len(checkpoints) == 1, "Expected one final checkpoint per run")
            with np.load(checkpoints[0], allow_pickle=False) as saved:
                weights, bias = saved["weights"], saved["bias"]
            require(weights.shape == (3744,) and bias.shape == (34,), "Unexpected MLP state dimensions")
            require(np.isfinite(weights).all() and np.isfinite(bias).all(), "Non-finite model state")
            w1, w2 = weights[:3680].reshape(115, 32), weights[3680:].reshape(32, 2)
            predicted = (np.maximum(x @ w1 + bias[:32], 0.0) @ w2 + bias[32:]).argmax(axis=1)
            with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(y, saved["y_true"])
                np.testing.assert_array_equal(predicted, saved["y_pred"])
            models += 1
    report.update(verified_final_models=models, parameter_count=3778)
    if manifest_files is not None:
        report["verified_manifest_files"] = manifest_files
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REFERENCE)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "code/data/dataset20000")
    parser.add_argument("--compare-reference", action="store_true", help="Also compare a complete rerun with the Table 6 reference")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.input_dir.resolve(), args.data_dir.resolve(), args.compare_reference), indent=2))


if __name__ == "__main__":
    main()
