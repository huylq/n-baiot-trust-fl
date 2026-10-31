"""Configuration and numerical summaries for the MLP experiment grid."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference/mlp"
sys.path.insert(0, str(ROOT / "code"))
from config import build_runtime_config

METRICS = [
    "accuracy", "f1_macro", "drop_vs_clean_fedavg_pp", "precision_10", "recall_10",
    "precision_cum", "recall_cum", "mean_ttd", "flag_fpr", "suppression_fpr",
    "mean_honest_suppressed_per_round", "final_malicious_weight_share",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_manifest_digest():
    contents = (ROOT / "data/dataset20000_manifest.csv").read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(contents).hexdigest()


def load_protocol(path=None):
    return json.loads((Path(path) if path else ROOT / "configs/MLP.json").read_text(encoding="utf-8"))


def runtime_config(protocol, data_dir=None):
    config = build_runtime_config()
    for target, origin in (("model", "training"), ("security", "security"),
                           ("defense", "defense"), ("split", "split")):
        config[target] = protocol[origin].copy()
    config["devices"] = protocol["clients"].copy()
    config["model"].update(architecture="mlp", hidden_units=protocol["layers"][1],
                            federated_rounds=protocol["rounds"])
    config["paths"]["raw_data_dir"] = str(data_dir or ROOT / "code/data/dataset20000")
    return config


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    require(bool(rows), f"No rows to write: {Path(path).name}")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_frame(rows):
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["case", "method", "rho", "malicious_count"], dropna=False)[METRICS]
    result = grouped.agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(filter(None, column)) for column in result.columns]
    result["seed_count"] = result["case"].map(frame.groupby("case")["seed"].nunique())
    return result


def add_clean_differences(rows):
    for row in rows:
        baseline = next((r for r in rows if r["seed"] == row["seed"]
                         and r["case"] == "clean_fedavg"), None)
        row["drop_vs_clean_fedavg_pp"] = ((baseline["accuracy"] - row["accuracy"]) * 100
                                         if baseline else None)


def check_manifest(directory):
    path = Path(directory) / "manifest.csv"
    require(path.is_file(), "Missing MLP file manifest")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len({r["file"] for r in rows}) == len(rows), "Duplicate manifest entries")
    expected = {r["file"] for r in rows}
    actual = {p.relative_to(directory).as_posix() for p in Path(directory).rglob("*")
              if p.is_file() and p != path}
    require(actual == expected, "MLP manifest file set differs")
    for row in rows:
        file = (Path(directory) / row["file"]).resolve()
        require(file.is_relative_to(Path(directory).resolve()), "Manifest path escapes input directory")
        require(file.stat().st_size == int(row["bytes"]) and sha(file) == row["sha256"],
                f"MLP checksum mismatch: {row['file']}")
    return len(rows)
