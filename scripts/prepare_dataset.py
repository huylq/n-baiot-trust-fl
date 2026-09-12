"""Construct and verify the fixed N-BaIoT subset used by the experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import DEVICES

RANDOM_SEED = 42
GAFGYT = ("combo", "junk", "scan", "tcp", "udp")
MIRAI = ("ack", "scan", "syn", "udp", "udpplain")


def verify_dataset(directory: Path) -> None:
    with (ROOT / "data" / "dataset20000_manifest.csv").open(encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    actual = {p.name for p in directory.glob("*.csv")}
    names = {row["file"] for row in expected}
    if actual != names:
        raise ValueError(f"Dataset filenames differ: missing={sorted(names - actual)}, extra={sorted(actual - names)}")
    failures = []
    for row in expected:
        path = directory / row["file"]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.stat().st_size != int(row["size_bytes"]) or digest != row["sha256"]:
            failures.append(path.name)
    if failures:
        raise ValueError(f"Dataset checksum mismatch in {len(failures)} files: {failures}")
    print(f"PASS: all {len(expected)} dataset files match the reference SHA-256 manifest.", flush=True)


def prepare_dataset(raw: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; existing data will not be overwritten.")
    expected = set(DEVICES)
    actual = {p.name for p in raw.iterdir() if p.is_dir()}
    if not expected.issubset(actual):
        raise ValueError(f"Missing device directories: {sorted(expected - actual)}")
    output.mkdir(parents=True, exist_ok=True)
    for device in sorted(DEVICES):
        source = raw / device
        has_mirai = (source / "mirai_attacks").is_dir()
        attack_count = 1000 if has_mirai else 2000
        jobs = [(source / "benign_traffic.csv", 10000, "benign")]
        jobs += [(source / "gafgyt_attacks" / f"{label}.csv", attack_count, f"gafgyt_{label}") for label in GAFGYT]
        if has_mirai:
            jobs += [(source / "mirai_attacks" / f"{label}.csv", attack_count, f"mirai_{label}") for label in MIRAI]
        for path, count, label in jobs:
            frame = pd.read_csv(path)
            if len(frame) < count:
                raise ValueError(f"Insufficient rows: {path.name}: {len(frame)} < {count}")
            sampled = frame.sample(n=count, random_state=RANDOM_SEED).reset_index(drop=True)
            destination = output / f"{device}_{count}sample_{label}.csv"
            # Fixed Windows line endings preserve the reference file hashes on all platforms.
            sampled.to_csv(destination, index=False, lineterminator="\r\n")
            print(f"[DATA] {destination.name}", flush=True)
    verify_dataset(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="Extracted full N-BaIoT device directories")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "code" / "data" / "dataset20000")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_dataset(args.output_dir)
    elif args.raw_dir is None:
        parser.error("--raw-dir is required unless --verify-only is set")
    else:
        prepare_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
