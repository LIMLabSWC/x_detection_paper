# The hippocampus enables structure learning without reward — analysis code

Code to reproduce the analyses and figures in Onih, Shen, Pentousi, Boboeva, Swarnabadran & Akrami,
*The hippocampus enables structure learning without reward* (Nature, 2026).

Archived release: Zenodo DOI (https://doi.org/10.5281/zenodo.24509218)

## Contents

| Folder | What it contains |
|---|---|
| `figures/` | One script per main / Extended Data figure (`final_fig_1.py` … `final_fig_5_b.py`, `final_ed_fig_1.py`) |
| `analysis_scripts/` | Pupil, behaviour, single-unit and population analysis functions (decoding, PSTHs, similarity, GLM) |
| `pupil_response_gen_scripts/` | Build the condition-wise pupil response files used by the figure scripts |
| `analysis_configs/` | YAML configuration for each analysis |
| `TimeSeries_clasification/` | Time-series classifier and clustering analyses |
| `matlab/` | **[to be added]** LDA state-space, cross-condition generalisation (CCGP) and full-space parallelism analyses (Fig. 5j–n, Extended Data Fig. 8) |
| `data/` | Data to run analsysis. can be downloaded from zenodo release

## 1. System requirements

- **Operating systems tested:** [Windows 11 / Ubuntu 22.04 / macOS 14]
- **Python** 3.10 with: numpy, panda, scipy 1.12.0, scikit-learn 1.7.2, statsmodels 0.15.0,
  matplotlib [x.y], pyGAM 0.12.0, MNE-Python 1.11.0, PyTables, PyYAML, joblib, tqdm,
  tensorflow [x.y] (only for `TimeSeries_clasification/`), and `xdetectioncore` [version; link to repository]
  **xdetectioncore** Can be installed using pip install xdetectioncore==0.1.7.6_b1
    (full list with versions in `requirements.txt`)
- **MATLAB** R2025a with the Statistics and Machine Learning Toolbox (for `matlab/`)
- Spike sorting (not needed to reproduce figures from processed data): SpikeInterface 0.102.3, Kilosort 4.0.18
- **Non-standard hardware:** none. A standard desktop computer (≥16 GB RAM) is sufficient.
  A GPU is needed only for Kilosort spike sorting.

## 2. Installation

```bash
git clone https://github.com/LIMLabSWC/x_detection_paper.git
cd x_detection_paper
conda create -n xdetection python=3.10
conda activate xdetection
pip install -r requirements.txt
```

Typical install time on a normal desktop computer: about [10] minutes.

## 3. Demo

```bash
cd figures
python final_fig_1.py --data ../demo_data
```

- **Input:** `demo_data/` contains pupil responses and trial tables for [one mouse / N sessions].
- **Expected output:** the pupil-response panels of Fig. 1e–g for the demo sessions, saved to `figures/output/`
  (an example of the expected output is in `demo_data/expected_output/`).
- **Expected run time** on a normal desktop computer: about [2] minutes.

## 4. Instructions for use

1. Set the data location in `config.yaml` (replace the lab-server paths with your local data folder).
2. Generate pupil response files with the scripts in `pupil_response_gen_scripts/`.
3. Run the figure scripts in `figures/`; each script states which data files it reads.
4. For Fig. 5j–n and Extended Data Fig. 8, run the MATLAB scripts in `matlab/` (see the header of each script).

## Reproduction instructions

Each figure panel is produced by the script named after the figure in `figures/`. The data needed to run them
on the full dataset are available from the corresponding author on reasonable request (see the paper's
Data availability statement). Source data for every figure are provided with the paper.

## Licence

MIT License — see `LICENSE`.

## Contact

Athena Akrami — athena.akrami@ucl.ac.uk