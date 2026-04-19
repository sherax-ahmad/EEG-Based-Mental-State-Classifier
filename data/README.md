# 📂 Dataset Setup Guide

This directory holds the EEG data used by the project.

```
data/
├── raw/           ← Original downloaded files (.edf, .csv)
│   ├── sleep-edf/
│   └── eegmmidb/
└── processed/     ← Preprocessed .npy arrays (auto-generated)
    ├── focus_data.npy
    ├── focus_labels.npy
    ├── stress_data.npy
    ├── stress_labels.npy
    ├── sleep_data.npy
    └── sleep_labels.npy
```

---

## 1. Sleep Stage Classification — Sleep-EDF Database

**Source:** PhysioNet  
**URL:** https://physionet.org/content/sleep-edfx/1.0.0/  
**Format:** EDF (European Data Format)  
**Subjects:** 197 (we use ≤ 20 for quick experiments)

### Download

```bash
# Via script (recommended)
python scripts/download_data.py --dataset sleep-edf --n-subjects 20

# Or manually with PhysioNet client
pip install wfdb
python -c "
import wfdb
wfdb.dl_database('sleep-edfx', 'data/raw/sleep-edf',
                 records=['SC4001E0-PSG', 'SC4001EC-Hypnogram'])
"
```

### Labels (AASM staging)

| Label | Stage | Description |
|-------|-------|-------------|
| 0 | W    | Wake |
| 1 | N1   | Light sleep |
| 2 | N2   | Core sleep |
| 3 | N3   | Deep / slow-wave sleep |
| 4 | REM  | Rapid Eye Movement |

---

## 2. Focus vs. Relaxed — EEG Motor Movement/Imagery Dataset

**Source:** PhysioNet  
**URL:** https://physionet.org/content/eegmmidb/1.0.0/  
**Format:** EDF  
**Subjects:** 109, 64-channel EEG

For focus/relaxed classification, use:
- **Run 01** (eyes open, relaxed → label 0)
- **Run 02** (eyes closed, relaxed → label 0)
- **Runs 04/08/12** (motor imagery task → label 1 for "focused")

### Download

```bash
python scripts/download_data.py --dataset eegmmidb --n-subjects 10
```

---

## 3. Stress Detection — DEAP Dataset

**Source:** Queen Mary University of London  
**URL:** https://www.eecs.qmul.ac.uk/mmv/datasets/deap/  
**Format:** CSV / MATLAB .dat  
**Subjects:** 32, 32-channel EEG + peripheral physiology

> ⚠️ DEAP requires manual registration on their website.  
> Download `data_preprocessed_python.zip`, unzip to `data/raw/deap/`.

Binary stress labels can be derived from the **arousal** dimension:
- Arousal ≥ 5 → Stress (label 1)
- Arousal < 5 → No Stress (label 0)

---

## Converting Raw → Processed NumPy Arrays

After downloading, run the preprocessing notebook or script:

```bash
# Using the notebook (recommended for visualization)
jupyter notebook notebooks/02_preprocessing.ipynb

# Or via the training script — it handles loading automatically if
# raw files are present and processed arrays don't exist yet.
```

---

## Quick Sanity Check

```python
import numpy as np

X = np.load("data/processed/sleep_data.npy")
y = np.load("data/processed/sleep_labels.npy")
print(f"Shape: {X.shape}")   # (n_epochs, n_channels, n_samples)
print(f"Labels: {np.unique(y, return_counts=True)}")
```
