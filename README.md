# Trust-Aware Federated Aggregation with Forensic Attribution on N-BaIoT

Research code for nine-device federated learning under label-flip attacks.
The experiments compare trust-based aggregation, robust aggregation baselines,
attacker attribution, and clean-data performance across the primary softmax
groups G0-G4 and an additional fixed-protocol MLP evaluation.

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

## Confusion matrices and class-wise metrics

Final-test classification results underlying manuscript Tables 4-5 are available
in [`reference/confusion_matrices/`](reference/confusion_matrices/):

- [`confusion_matrices.csv`](reference/confusion_matrices/confusion_matrices.csv):
  75 matrices identified by configuration, seed and device.
- [`classwise_metrics.csv`](reference/confusion_matrices/classwise_metrics.csv):
  150 rows with benign/malicious precision, recall, F1 and support.
- [`summary.csv`](reference/confusion_matrices/summary.csv): seven configuration
  summaries with means and sample standard deviations over five seeds.

Table 4 covers centralized, local-only, clean FedAvg (`S1`) and the clean proposed
method (`S1_trust`). Table 5 covers the proposed method (`S4`) with one, two or
three attackers and label-flipping probability 1.0. Seeds are 0-4. Centralized,
FedAvg and proposed models each have one matrix per configuration and seed.
Local-only has nine device matrices per seed, giving 45 of the 75 matrices.

Matrix rows are true labels and columns are predictions, ordered `benign,
malicious`. With malicious as the positive class, the matrix is
`[[TN, FP], [FN, TP]]`. Metrics use scale 0-1; multiply by 100 for percentages.
For local-only summaries, device metrics are averaged within each seed before
calculating means and sample standard deviations over seeds. F1 computed from
summed device matrices is not substituted for that average. Accuracy and
macro-F1 differ before rounding, but their means and standard deviations
coincide at the three-decimal percentage precision used in Tables 4-5.

Regenerate these files with the included dataset and existing experiment code:

```sh
python scripts/export_confusion_matrices.py
```

The script reruns only the configurations in Tables 4-5, using 50 federated
rounds and 20 epochs for centralized/local-only models. It exports the final-test
confusion matrices and checks the recomputed per-seed metrics against
`reference/tables/all_seed_runs.csv` with
absolute tolerance 1e-12. Exports, audit logs and scalers are saved separately
under `outputs/confusion_matrices/`. Existing output directories are refused;
use `--output-dir PATH` for a rerun. `--data-dir PATH` selects another verified
copy of the dataset. No previously saved models are required.

## Additional MLP evaluation

The nonlinear evaluation uses a 115-input, 32-unit ReLU hidden layer and two
softmax outputs (3,778 parameters). It retains the primary data splits, pooled
training-only standardization, SGD settings and security parameters without
model-specific tuning. The fixed protocol is in [`configs/MLP.json`](configs/MLP.json).
Five seeds and 50 rounds are used for each of eight configurations:

- Clean FedAvg and S4.
- FedAvg, norm-clipped mean and S4 with three fixed attackers at each of
  label-flipping probabilities 0.2 and 1.0.

The 40-run results underlying manuscript Table 6 are in
[`reference/mlp/`](reference/mlp/). They include per-seed metrics, summary means
and sample standard deviations, 40 final-test confusion matrices, 80 class-wise
metric rows, and attribution counts through rounds 10 and 50. The directory
also contains all 40 final checkpoints, saved test predictions, original audit
chains, per-run configurations and the five training-data scalers. Archived
runtime paths are relative to each run directory. The data are unchanged from
the primary experiments. File hashes are recorded in `manifest.csv`.

Verify the archived evidence without training:

```sh
python scripts/verify_mlp.py
```

This checks the file manifest and dataset hashes, rebuilds the test sets and
scalers, independently evaluates all checkpoints, validates the audit chains,
replays 6,750 S4 client-round strike decisions and checks the supplied numerical
tables. It does not establish that attribution is correct merely because a hash
chain is valid. Cosine values are read from the logs because intermediate client
update vectors are not archived.

Regenerate the tables and diagnostics without training:

```sh
python scripts/export_mlp_results.py
```

Exports go to `outputs/mlp_exports/`. The exporter recomputes classification
metrics from saved predictions and attribution from the logs. It also exports
all S4 client-round traces, clean false-flag diagnostics and histories of missed
attackers. These support the reported clean-run false flags and the two missed
attackers in seed 0 at probability 1.0, alongside the successful detections.

Run the full 40-run experiment independently:

```sh
python scripts/run_mlp.py
python scripts/verify_mlp.py --input-dir outputs/mlp --compare-reference
python scripts/export_mlp_results.py --input-dir outputs/mlp --output-dir outputs/mlp_rerun_exports
```

The default is one worker. Add `--workers 2` or another positive count to run
seeds concurrently if memory permits. Each worker uses one BLAS thread. Use
`--data-dir PATH` with the runner and verifier for another verified dataset copy.
The runner and exporter refuse existing output directories. Select a new
`--output-dir PATH` for a rerun; the primary G0-G4 outputs are separate.

A short execution check, which is not a manuscript result:

```sh
python scripts/run_mlp.py --seeds 0 --rounds 8 --cases clean_fedavg clean_s4 --output-dir outputs/mlp_smoke
python scripts/verify_mlp.py --input-dir outputs/mlp_smoke
```

Accuracy and F1 use scale 0-1 in the CSV files. Table 6 expresses accuracy as a
percentage. Attribution precision and recall use distinct clients ever flagged,
not traffic samples. Mean detection time includes detected attackers only.
Cumulative recall 1.0 does not imply continuous exclusion, and final malicious
weight share is an aggregation weight, not a measure of attack damage. Clean
S4 false flags, missed attackers and residual weights expose model dependence
under the unchanged security parameters.

## Client attribution evidence

Attribution evidence for manuscript Table 5 is available in
[`reference/client_attribution/`](reference/client_attribution/). It covers S4
with one, two or three attackers, label-flipping probability 1.0, seeds 0-4,
and 50 rounds:

- [`audit/`](reference/client_attribution/audit/): 15 archived G3 experiment logs,
  preserved byte-for-byte, with SHA-256 hashes in
  [`audit_manifest.csv`](reference/client_attribution/audit_manifest.csv).
- [`configuration.json`](reference/client_attribution/configuration.json):
  device identities, nested attacker sets, training parameters and security rules.
- [`client_round_traces.csv`](reference/client_attribution/client_round_traces.csv):
  6,750 client-round records with ground truth, cosine, norm z-score, violations,
  strikes and flags.
- [`per_seed_summary.csv`](reference/client_attribution/per_seed_summary.csv):
  15 rows with cumulative TP/FP/FN/TN, precision and recall through rounds 10 and
  50, mean time to detection, and final cumulative Flag FPR.
- [`summary.csv`](reference/client_attribution/summary.csv): means and sample
  standard deviations across the five seeds for each attacker count.
- [`false_positive_events.csv`](reference/client_attribution/false_positive_events.csv):
  five client-seed cases, listing the first false flag, violation rounds within
  its five-round window, and all flagged rounds.
- [`figures/`](reference/client_attribution/figures/): PDF/PNG plots for all three
  attacker counts. Upper panels show cosine means and min-max ranges across
  clients and seeds. Gray shading denotes warm-up and the dashed line is the
  cosine violation threshold. Lower panels show the fraction of seeds flagged
  for each client and round. Asterisks mark attackers.

After five warm-up rounds, a violation is recorded when cosine is below -0.1
or the robust log-norm z-score exceeds 3.5. At least three violations within
a five-round window trigger a flag. The three violations need not be consecutive.
Flags can clear when older violations leave the window. Ground truth is used
to score attribution, not to make detection decisions.

Every attacker is first flagged at round 8 in all 15 runs. With two attackers,
no benign client is flagged over 50 rounds. With one attacker, SimpleHome 1002
is falsely flagged in seeds 0 and 1, first at rounds 17 and 14. With three
attackers, Samsung and Ecobee are falsely flagged in seed 2, first at rounds 42
and 45, and Samsung in seed 3 at round 43. These false flags follow benign cosine
violations, with no norm violations in their triggering windows. Consequently,
mean cumulative precision is 0.80, 1.00 and 0.87 for one, two and three attackers.

Counts through a horizon are based on the union of distinct clients ever
flagged through that round. A later flag removal does not remove that client
from cumulative counts. Thus, cumulative recall 1.0 does not mean every attacker
remains flagged at every round. Flag FPR is the fraction of distinct benign
clients ever flagged, not the fraction of client-round decisions. All false
flags occur after round 10, so detection-window precision remains 1.0.
The two-attacker cases in Fig. 1 and Table 5 use the same protocol and seeds
and should not be treated as independent replications.

Recreate the CSV files and figures from the included archived logs:

```sh
python scripts/export_client_attribution.py
```

This command does not train models. It verifies the archived file hashes and
audit hash chains, recomputes norm z-scores and all 6,750 strike/flag decisions,
and compares the per-seed attribution metrics with the supplied reference
tables. A verification report and regenerated artifacts are written under
`outputs/client_attribution/`. Existing output directories are refused. Use
`--output-dir PATH` to select a new destination. `--audit-dir PATH` accepts a
directory containing the same 15 log filenames from a rerun and applies the
same chain, decision and metric checks. Archived-file manifest checks apply
only to the included audit directory.

The archived logs reproduce the supplied Table 5 attribution metrics. Cosine
values are read from the logs because raw update vectors are not included.
Full experiment reproduction remains available via
`scripts/run_grid.py --groups G3`.

## Files and outputs

- `code/src/`: model, data splits, attacks, proposed method, baselines, metrics,
  audit chains, and exports; `code/config.py`: seeds and hyperparameters.
- `configs/G0.json` ... `G4.json`: experiment-group configurations.
- `code/data/dataset20000/`: fixed experiment subset; `LICENSE-DATA.md`: source,
  sampling provenance and dataset-specific license.
- `scripts/`: dataset preparation, experiment execution, verification and artifact export,
  including the Table 4-5 confusion-matrix exporter.
- `reference/`: numerical reference tables/workbooks and code-generated PDF/PNG figures.
- `reference/confusion_matrices/`: per-run confusion matrices, class-wise metrics
  and configuration summaries for manuscript Tables 4-5.
- `reference/client_attribution/`: archived logs, attribution traces, per-seed
  counts and supporting figures for manuscript Table 5.
- `reference/mlp/`: Table 6 metrics, MLP checkpoints, predictions, scalers,
  audit chains and attribution diagnostics.
- `configs/MLP.json` and `scripts/{run_mlp,verify_mlp,export_mlp_results}.py`:
  fixed nonlinear protocol, training, verification and numerical exports.
- `outputs/<group>/runs/<run>/`: configuration, logs, audit chains, models,
  workbook and diagnostic figures; `outputs/runs.json`: selected run index.
- `paper/{tables,excel,figures}/`: newly generated tables, workbooks, PDF/PNG figures.
- [Reproducibility notes](docs/reproducibility.md): protocol, metric definitions,
  expected row counts, figure mapping and execution constraints.

## License

Dataset terms are provided in `LICENSE-DATA.md`. No source-code license has
been specified.
