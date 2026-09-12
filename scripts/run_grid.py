"""Run isolated experiment groups with a relocatable result index."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["G0", "G1", "G2", "G3", "G4"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "code" / "data" / "dataset20000")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()
    if args.rounds < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Rounds must be positive and seeds must be unique.")
    if len(set(args.groups)) != len(args.groups):
        parser.error("Groups must be unique.")
    from prepare_dataset import verify_dataset
    verify_dataset(args.data_dir)
    output = args.output_dir.resolve()
    index_path = output / "runs.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    for group in args.groups:
        if group in index or (output / group).exists():
            parser.error(f"{group} already exists in {output}; choose a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        group_dir = output / group
        group_dir.mkdir()
        config = json.loads((ROOT / "configs" / f"{group}.json").read_text(encoding="utf-8"))
        config["paths"] = {
            "raw_data_dir": str(args.data_dir.resolve()),
            "output_root": str(group_dir),
            "runs_root": str(group_dir / "runs"),
        }
        config_path = group_dir / "override.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        command = [sys.executable, "-u", str(ROOT / "code" / "run_experiments.py"),
                   "--config", str(config_path), "--seeds", *map(str, args.seeds),
                   "--rounds", str(args.rounds)]
        index[group] = {"status": "running", "seeds": args.seeds, "rounds": args.rounds}
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with (group_dir / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT / "code", env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                index[group]["status"] = "interrupted"
                index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
                raise
        candidates = list((group_dir / "runs").glob("run_*"))
        complete = return_code == 0 and len(candidates) == 1 and (candidates[0] / "logs" / "run_summary.json").is_file()
        index[group]["status"] = "complete" if complete else "failed"
        if len(candidates) == 1:
            index[group]["run_root"] = candidates[0].relative_to(output).as_posix()
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        if not complete:
            raise SystemExit(f"{group} failed; see {group_dir / 'console.log'}")
    print(f"[INDEX] {index_path}")


if __name__ == "__main__":
    main()
