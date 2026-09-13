# Reproducibility notes

## Protocol

The subset contains 180,000 rows across nine devices and 115 numeric features.
Each device contributes 10,000 benign and 10,000 malicious examples. Sampling
uses pandas `DataFrame.sample(n=..., random_state=42)` independently for every
input CSV, without replacement. All 89 files are included under
`code/data/dataset20000/` and identified by the checked-in SHA-256 manifest.
Git preserves their bytes, including line endings. The subset is distributed
under CC BY 4.0; attribution and sampling provenance are in `LICENSE-DATA.md`.

Each experiment seed (0, 1, 2, 3, 4) defines stratified per-device 70/15/15
train/validation/test splits. Pooled feature scaling uses training data only.
Label flipping affects training labels of configured attackers only. Ecobee
remains honest. Training uses a two-class softmax model, learning rate 0.01,
batch size 64, no L2 penalty, 50 federated rounds and one local epoch per round.
Centralized and local-only baselines use 20 epochs. All remaining parameters
are defined in `code/config.py`; group selection is in `configs/G0.json` through
`configs/G4.json` and the grid definition is in `code/src/federated.py`.

| Group | Scope | Rows with five seeds |
| --- | --- | ---: |
| G0 | Centralized, local-only, clean FedAvg, clean trust | 20 |
| G1 | Clean FedAvg and ten attacked methods, rho=1, m=2 | 55 |
| G2 | Six rho values, three methods, m=3; plus clean baseline | 95 |
| G3 | m=1,2,3; three methods, rho=1 | 45 |
| G4 | Full method and four variants, rho=1, m=3 | 25 |

The complete grid has 240 per-seed result rows, including repeated configurations
across groups. These are not 240 independent experimental conditions.

## Reference and comparison

`reference/tables/` and `reference/excel/` contain the numerical comparison
targets. `reference/figures/` contains PDF/PNG figures generated from experiment
results by the figure code. Numerical verification uses result records, not
pixels or PDF file hashes. New exports are written separately to `paper/`.

`scripts/verify_results.py` matches experiment keys and compares numeric values
with absolute tolerance 1e-12. It reports missing/extra rows and any diagnostic
values without a reference value. It does not claim that every platform
or BLAS implementation produces bit-identical results. The dependency lock was
validated with Python 3.11 on Windows; other platforms require validation.
PDF metadata, raster rendering and file timestamps are not expected to match.

## Tables and figures

Run `python scripts/build_artifacts.py` after completing G0-G4 with all five
seeds and 50 rounds. The command exports 15 CSV tables, five Excel workbooks
and seven figures in both PDF and PNG formats. It also records the input run
paths in the generated output README. No manuscript text is generated.

The figure filenames identify plots, not their numbering in the manuscript:

| Figure file stem | Content |
| --- | --- |
| `fig1_rho_boundary` | Label-flipping probability sweep (G2) |
| `fig2_main_comparison` | Main method comparison (G1) |
| `fig3_trust_trajectories` | Representative trust trajectories (G1, seed 0) |
| `fig4_attribution_rounds` | Cumulative attribution across rounds (G1/G4) |
| `fig5_g0_diagnostics` | Honest-client update diagnostics (G0) |
| `fig6_ablation` | Mechanism variants (G4) |
| `fig7_malicious_count` | Nested attacker-set comparison (G3) |

Accuracy is shown as a percentage; `pp` denotes percentage points. Accuracy
error bars show mean +/- one sample standard deviation across five seeds,
not confidence intervals. Figure annotations and plot ranges target the
documented experiment grid; check them when changing the configuration.

## Metric interpretation and execution constraints

- Mean time to detection averages detected attackers only; it must be read with
  recall. Cumulative attribution uses strike flags, not every zero-weight client.
  Suppression is a separate cost-of-defense diagnostic. A flag is not a permanent
  quarantine because strikes can expire.
- The detection-window table uses rounds 1-10, not the entire training horizon.
  Full-run cumulative precision can therefore differ from detection-window precision.
- S3 can reject all updates after warm-up and retain the preceding model. Its
  high clean-test accuracy does not establish successful continued learning.
- The model writer names checkpoints by scenario and seed, without
  rho or attacker count. G2/G3 consequently overwrite earlier checkpoints
  within a run. Per-configuration metrics and audit chains remain distinct.
  Use logs/workbooks for the full grid,
  not the number of model files as a completion check. Local-only models are
  not exported. Re-evaluating a checkpoint also requires its training-data scaler.
- Use `scripts/build_artifacts.py`, rather than invoking individual builders,
  so input paths are taken from the selected `runs.json` index.
