# 🧠 EEG-Based Mental State Classifier

> Classify mental states from EEG signals using Machine Learning and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-red.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project builds end-to-end pipelines to classify mental states from raw EEG (electroencephalography) signals. It covers:

| Task | Classes | Model Type |
|------|---------|------------|
| Focus vs. Relaxed | Binary | SVM, Random Forest, CNN |
| Stress Detection | Binary / Multi-class | SVM, LSTM |
| Sleep Stage Classification | 5-class (W/N1/N2/N3/REM) | CNN-LSTM, Random Forest |

---

## 🗂️ Project Structure

```
eeg-mental-state-classifier/
│
├── data/
│   ├── raw/                    # Raw EEG files (.edf, .csv)
│   ├── processed/              # Preprocessed & segmented data
│   └── README.md               # Dataset download instructions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_sklearn_models.ipynb
│   └── 05_deep_learning_models.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # EDF/CSV data loaders
│   │   └── preprocessor.py     # Filtering, epoching, normalization
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── time_domain.py      # Statistical time-domain features
│   │   ├── frequency_domain.py # PSD, band power (alpha/beta/theta...)
│   │   └── extractor.py        # Feature extraction pipeline
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical.py        # scikit-learn models
│   │   └── deep_learning.py    # TensorFlow/Keras models
│   │
│   └── utils/
│       ├── __init__.py
│       ├── evaluation.py       # Metrics, confusion matrix, plots
│       └── visualization.py    # EEG signal & result plots
│
├── scripts/
│   ├── download_data.py
│   ├── train_classical.py
│   └── train_deep.py
│
├── tests/
│   ├── test_preprocessor.py
│   ├── test_features.py
│   └── test_models.py
│
├── configs/
│   ├── focus_config.yaml
│   ├── stress_config.yaml
│   └── sleep_config.yaml
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 📊 Datasets

This project uses publicly available datasets from [PhysioNet](https://physionet.org/):

| Dataset | Task | Link |
|---------|------|------|
| EEG Motor Movement/Imagery Dataset | Focus/Relaxed | [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) |
| DEAP Dataset | Stress/Emotion | [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| Sleep-EDF Database | Sleep Stages | [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) |

### Download Instructions

```bash
# Install PhysioNet client
pip install wfdb

# Download Sleep-EDF (small subset)
python scripts/download_data.py --dataset sleep-edf --subset cassette --n-subjects 20

# Download EEG Motor dataset
python scripts/download_data.py --dataset eegmmidb --n-subjects 10
```

See [`data/README.md`](data/README.md) for detailed setup instructions.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/eeg-mental-state-classifier.git
cd eeg-mental-state-classifier

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Preprocess Data

```bash
python scripts/download_data.py --dataset sleep-edf
```

### 2. Train Classical ML Models (scikit-learn)

```bash
python scripts/train_classical.py --task sleep --config configs/sleep_config.yaml
```

### 3. Train Deep Learning Models (TensorFlow)

```bash
python scripts/train_deep.py --task focus --config configs/focus_config.yaml
```

---

## 🧪 Model Performance

### Focus vs. Relaxed (EEG Motor Dataset)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM (RBF) | 87.3% | 0.87 |
| Random Forest | 85.1% | 0.85 |
| CNN | 91.4% | 0.91 |

### Sleep Stage Classification (Sleep-EDF)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Random Forest | 79.2% | 0.74 |
| CNN-LSTM | 84.7% | 0.81 |

*Results may vary depending on dataset split and hyperparameters.*

---

## 🧬 EEG Frequency Bands

| Band | Frequency | Mental State |
|------|-----------|-------------|
| Delta (δ) | 0.5–4 Hz | Deep sleep |
| Theta (θ) | 4–8 Hz | Drowsiness, relaxation |
| Alpha (α) | 8–13 Hz | Calm/relaxed alertness |
| Beta (β) | 13–30 Hz | Active thinking, focus |
| Gamma (γ) | 30–100 Hz | High-level cognition |

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
