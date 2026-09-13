# Trust-Aware Federated Aggregation with Forensic Attribution on N-BaIoT

Research code for nine-device federated learning under label-flip attacks.
The experiments compare trust-based aggregation, robust aggregation baselines,
attacker attribution, and clean-data performance across groups G0-G4.

## Installation

Use Python 3.11 and run commands from the repository root. CPU execution is sufficient.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

Use the pinned dependencies in `requirements-lock.txt` for reproduction.

## Included dataset

The exact experiment subset is included in `code/data/dataset20000/`: 89 CSV
files, 180,000 samples across nine devices, approximately 200 MiB. No separate
dataset download or preprocessing is required to run the experiments.

Verify the included files before running:

```sh
python scripts/prepare_dataset.py --verify-only
```

This compares all files with `data/dataset20000_manifest.csv`. Git preserves
their original line endings so the SHA-256 checks also work after checkout.
The subset derives from N-BaIoT by Meidan et al. (2018), distributed by UCI
under CC BY 4.0. See [dataset attribution and license](LICENSE-DATA.md);
this license applies to the dataset, not the source code.

## Optional: rebuild the subset from raw data

Download [N-BaIoT from UCI](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot)
([dataset DOI](https://doi.org/10.24432/C5RC8J)). Extract each device directory
and its attack archives, including the nested RAR files, into CSV directories:

```text
data/full-raw/
  Danmini_Doorbell/
    benign_traffic.csv
    gafgyt_attacks/{combo,junk,scan,tcp,udp}.csv
    mirai_attacks/{ack,scan,syn,udp,udpplain}.csv
  Ecobee_Thermostat/
  Ennio_Doorbell/                       # no Mirai directory
  Philips_B120N10_Baby_Monitor/
  Provision_PT_737E_Security_Camera/
  Provision_PT_838_Security_Camera/
  Samsung_SNH_1011_N_Webcam/            # no Mirai directory
  SimpleHome_XCS7_1002_WHT_Security_Camera/
  SimpleHome_XCS7_1003_WHT_Security_Camera/
```

All devices follow the first device's layout except the two without Mirai.
Do not create empty Mirai directories for those devices.

```sh
python scripts/prepare_dataset.py --raw-dir data/full-raw --output-dir data/dataset20000
```

This samples 10,000 benign rows plus 10,000 attack rows per device with seed 42:
1,000 per attack type for ten-type devices, or 2,000 per type for five-type
devices. This optional command writes a separate copy to `data/dataset20000/`
and checks its SHA-256 hashes against `data/dataset20000_manifest.csv`. Use an
empty output directory; it will not overwrite the included subset. Full raw
data and this optional rebuilt copy are excluded from Git. Keep several GB of
free RAM and sufficient disk space if rebuilding from the extracted raw data.

## Run experiments

Smoke test (one seed, five federated rounds; not a paper result):

```sh
python scripts/run_grid.py --groups G0 --seeds 0 --rounds 5 --output-dir outputs/smoke
```

Representative experiment (G1, one seed, all 50 rounds):

```sh
python scripts/run_grid.py --groups G1 --seeds 0 --output-dir outputs/example
python scripts/verify_results.py --index outputs/example/runs.json --allow-partial
```

Full paper grid: five seeds **0, 1, 2, 3, 4** and **50 federated rounds**:

```sh
python scripts/run_grid.py
python scripts/verify_results.py
python scripts/build_artifacts.py
```

Groups run sequentially in isolated processes. Add `--data-dir PATH` to use a
verified dataset elsewhere. Existing group outputs are never overwritten;
choose a new `--output-dir` for a rerun. Use its `runs.json` with the verification
and artifact-generation commands. Figure generation requires all G0-G4 results.

For results saved outside the default output directory:

```sh
python scripts/verify_results.py --index outputs/rerun/runs.json
python scripts/build_artifacts.py --index outputs/rerun/runs.json --output-dir exports/paper
```

`build_artifacts.py` generates tables, Excel workbooks and PDF/PNG figures from
completed results. It does not generate manuscript text. The export directory
must be empty and its final path component must be `paper`.

Tests:

```sh
cd code
python -m unittest discover -s tests -v
cd ..
python -m unittest discover -s tests -v
```

## Files and outputs

- `code/src/`: model, data splits, attacks, proposed method, baselines, metrics,
  audit chains, and exports; `code/config.py`: seeds and hyperparameters.
- `configs/G0.json` ... `G4.json`: experiment-group configurations.
- `code/data/dataset20000/`: fixed experiment subset; `LICENSE-DATA.md`: source,
  sampling provenance and dataset-specific license.
- `scripts/`: dataset preparation, experiment execution, verification and artifact export.
- `reference/`: numerical reference tables/workbooks and code-generated PDF/PNG figures.
- `outputs/<group>/runs/<run>/`: configuration, logs, audit chains, models,
  workbook and diagnostic figures; `outputs/runs.json`: selected run index.
- `paper/{tables,excel,figures}/`: newly generated tables, workbooks, PDF/PNG figures.
- [Reproducibility notes](docs/reproducibility.md): protocol, metric definitions,
  expected row counts, figure mapping and execution constraints.

## License

Dataset terms are provided in `LICENSE-DATA.md`. No source-code license has
been specified.
