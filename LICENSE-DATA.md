# Dataset license and attribution

## Scope

The 89 CSV files in `code/data/dataset20000/` are a sampled subset of
**detection_of_IoT_botnet_attacks_N_BaIoT**, released through the UCI Machine
Learning Repository in 2018. They are not newly collected traffic from this
project.

Credit: Yair Meidan, Michael Bohadana, Yael Mathov, Yisroel Mirsky,
Dominik Breitenbacher, A. Shabtai and Y. Elovici, the authors of the associated
N-BaIoT study. See the UCI record for dataset contributor and citation metadata.

- Source: [N-BaIoT at UCI](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot).
- Dataset DOI: [10.24432/C5RC8J](https://doi.org/10.24432/C5RC8J).
- Associated study: *N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using
  Deep Autoencoders*, Meidan et al., 2018, IEEE Pervasive Computing.

## License

UCI identifies this dataset as licensed under **Creative Commons Attribution
4.0 International (CC BY 4.0)**. That license applies to the redistributed
subset; it does not select a license for this repository's source code.

- [CC BY 4.0 license summary](https://creativecommons.org/licenses/by/4.0/).
- [Full legal terms](https://creativecommons.org/licenses/by/4.0/legalcode).

When redistributing or adapting the data, retain attribution and the license
link, identify changes, and comply with the full license terms. No endorsement
by the original contributors or UCI is implied. The license provides no
warranties; see its disclaimer and limitation-of-liability provisions.

## Changes from the original dataset

The experiment subset samples rows without replacement from each original CSV
using pandas `DataFrame.sample(n=..., random_state=42)` independently per file.
Each of the nine devices contributes 10,000 benign and 10,000 attack samples:
1,000 per attack file for devices with ten attack types, or 2,000 per attack
file for devices with five types. Total: 180,000 rows in 89 CSVs.

The sampled rows are reordered by sampling, CSV files are renamed to include
device, sample count and attack label, and numeric values are re-serialized by
pandas without an index. No feature engineering or attack-label flipping is
performed in this dataset preparation step. Training splits, scaling and
label-flip attacks are applied separately by the experiment code.

`scripts/prepare_dataset.py` implements this procedure.
`data/dataset20000_manifest.csv` records filenames, byte counts and SHA-256
hashes. The included subset has been matched byte-for-byte to the experiment
input and to an independently regenerated copy using the pinned environment.
Full raw N-BaIoT data is not included.
