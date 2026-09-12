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

`reference/tables/` and `reference/excel/` contain the supplied numeric results.
`reference/figures/` preserves both the original PNGs and the manually edited
PDFs for comparison. Neither image format is used as numeric ground truth.
These are intentionally distinct from newly generated `paper/` outputs.

`scripts/verify_results.py` matches experiment keys and compares numeric values
with absolute tolerance 1e-12. It reports missing/extra rows and supplemental
diagnostics absent from earlier exports. It does not claim that every platform
or BLAS implementation produces bit-identical results. The dependency lock was
validated with Python 3.11 on Windows; other platforms require validation.
PDF metadata, raster rendering and file timestamps are not expected to match.

## Preserved implementation limitations

- Mean time to detection averages detected attackers only; it must be read with
  recall. Cumulative attribution uses strike flags, not every zero-weight client.
  Suppression is a separate cost-of-defense diagnostic. A flag is not a permanent
  quarantine because strikes can expire.
- The detection-window table uses rounds 1-10, not the entire training horizon.
  Full-run cumulative precision can therefore differ from detection-window precision.
- S3 can reject all updates after warm-up and retain the preceding model. Its
  high clean-test accuracy does not establish successful continued learning.
- The original model writer names checkpoints by scenario and seed, without
  rho or attacker count. G2/G3 consequently overwrite earlier checkpoints
  within a run. Per-configuration metrics and audit chains remain distinct.
  This release preserves that behavior; use logs/workbooks for the full grid,
  not the number of model files as a completion check. Local-only models are
  not exported. Re-evaluating a checkpoint also requires its training-data scaler.
- Original table/figure builders contain historical run identifiers. The
  documented `scripts/build_paper.py` command supplies current paths and adapts
  the G1 S4 audit filename; it does not alter table calculations or figure data.
- Figures contain fixed annotations tied to the supplied five-seed experiment.
  In Figure 6, the inherited phrase `one-seed collapse` is inconsistent with
  the per-seed values: seeds 1 and 4 have the lower mean-reference accuracies.
  The x-axis minimum of 98.2% also clips part of that error bar (mean about
  98.523%, sample standard deviation about 0.938 percentage points).
  These presentation issues are retained for author review, not silently corrected.
- Manually edited reference PDFs cannot be reconstructed byte-for-byte by the
  source figure generator. Numeric agreement and label/layout agreement are
  separate checks. The figure builder requires the complete five-seed grid.
- The current exports include an additional honest-suppression diagnostic and
  G1 clean-baseline significance comparisons absent from some older workbooks.
  Such additions are not differences in their shared metric values.

## Scope

This repository accompanies the supplied experiment implementation. Manuscript
drafting tools, thesis-report material, local audit utilities and full raw data
are not part of the public source package. The fixed experiment subset is
included. Dataset preparation preserves the
sampling procedure that recreates the supplied subset; it is not part of model
training or a change to the federated algorithm.

A code license and final paper citation must be selected by the authors before
public release. No license is inferred from the dataset's separate license.
