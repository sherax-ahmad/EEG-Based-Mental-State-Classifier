"""
Train classical ML models (SVM, Random Forest, etc.) on EEG data.

Usage:
    python scripts/train_classical.py --task focus --config configs/focus_config.yaml
    python scripts/train_classical.py --task sleep --config configs/sleep_config.yaml
"""

import argparse
import sys
import os
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import EEGDataLoader
from src.data.preprocessor import EEGPreprocessor
from src.features.extractor import EEGFeatureExtractor
from src.models.classical import EEGClassicalClassifier
from src.utils.evaluation import plot_confusion_matrix, compare_models


TASK_LABELS = {
    "focus":  ["Relaxed", "Focused"],
    "stress": ["No Stress", "Stress"],
    "sleep":  ["Wake", "N1", "N2", "N3", "REM"],
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict, task: str):
    """Load and preprocess data based on config."""
    loader = EEGDataLoader(data_dir=cfg.get("data_dir", "data/processed"))
    sfreq = cfg.get("sfreq", 256)
    preprocessor = EEGPreprocessor(
        sfreq=sfreq,
        lowcut=cfg.get("lowcut", 0.5),
        highcut=cfg.get("highcut", 45.0),
        epoch_length=cfg.get("epoch_length", 4.0),
        overlap=cfg.get("overlap", 0.5),
        amplitude_threshold=cfg.get("amplitude_threshold", 100.0),
    )
    extractor = EEGFeatureExtractor(sfreq=sfreq)

    data_file = cfg.get("data_file", f"data/processed/{task}_data.npy")
    labels_file = cfg.get("labels_file", f"data/processed/{task}_labels.npy")

    if os.path.exists(data_file) and os.path.exists(labels_file):
        print(f"Loading preprocessed data from {data_file}")
        epochs = np.load(data_file)
        labels = np.load(labels_file)
    else:
        raise FileNotFoundError(
            f"Data files not found: {data_file}, {labels_file}\n"
            "Run scripts/download_data.py first."
        )

    X = extractor.transform(epochs)
    return X, labels, sfreq


def main():
    parser = argparse.ArgumentParser(description="Train classical EEG classifiers")
    parser.add_argument("--task",   type=str, default="focus", choices=["focus", "stress", "sleep"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=["svm", "random_forest", "lda"],
                        help="Models to train (svm, random_forest, gradient_boosting, knn, lda)")
    parser.add_argument("--cv",     type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--output-dir", type=str, default="models/classical")
    args = parser.parse_args()

    # Config
    config_path = args.config or f"configs/{args.task}_config.yaml"
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    print(f"\n{'='*60}")
    print(f"Task: {args.task.upper()}")
    print(f"Models: {args.models}")
    print(f"{'='*60}")

    # Data
    X, y, sfreq = load_data(cfg, args.task)
    class_names = TASK_LABELS.get(args.task, [str(i) for i in range(len(np.unique(y)))])
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train & evaluate each model
    all_results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in args.models:
        print(f"\n--- {model_name.upper()} ---")
        clf = EEGClassicalClassifier(model_name=model_name)
        clf.train(X_train, y_train)

        cv_result = clf.cross_validate(X_train, y_train, cv=args.cv)
        test_result = clf.evaluate(X_test, y_test, target_names=class_names)
        all_results[model_name] = test_result

        # Save
        save_path = os.path.join(args.output_dir, f"{args.task}_{model_name}.pkl")
        clf.save(save_path)

        # Confusion matrix
        plot_confusion_matrix(
            test_result["confusion_matrix"],
            class_names=class_names,
            title=f"{model_name.upper()} — {args.task.capitalize()} Confusion Matrix",
            save_path=os.path.join(args.output_dir, f"{args.task}_{model_name}_cm.png"),
        )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, result in all_results.items():
        print(f"  {name:<20} Accuracy={result['accuracy']:.4f}  F1={result['f1_macro']:.4f}")

    compare_models(
        all_results,
        metric="accuracy",
        save_path=os.path.join(args.output_dir, f"{args.task}_model_comparison.png"),
    )


if __name__ == "__main__":
    main()
