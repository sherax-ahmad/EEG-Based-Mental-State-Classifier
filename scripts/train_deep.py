"""
Train deep learning models (CNN, LSTM, CNN-LSTM) on EEG data.

Usage:
    python scripts/train_deep.py --task focus --model cnn
    python scripts/train_deep.py --task sleep --model cnn_lstm --epochs 100
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
from src.models.deep_learning import (
    build_cnn,
    build_lstm,
    build_cnn_lstm,
    EEGDeepClassifier,
)
from src.utils.evaluation import (
    plot_confusion_matrix,
    plot_training_history,
    plot_roc_curves,
)


TASK_LABELS = {
    "focus":  ["Relaxed", "Focused"],
    "stress": ["No Stress", "Stress"],
    "sleep":  ["Wake", "N1", "N2", "N3", "REM"],
}

MODEL_BUILDERS = {
    "cnn":      build_cnn,
    "lstm":     build_lstm,
    "cnn_lstm": build_cnn_lstm,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_epochs(cfg: dict, task: str):
    """Load preprocessed epoch arrays."""
    data_file   = cfg.get("data_file",   f"data/processed/{task}_data.npy")
    labels_file = cfg.get("labels_file", f"data/processed/{task}_labels.npy")

    if not (os.path.exists(data_file) and os.path.exists(labels_file)):
        raise FileNotFoundError(
            f"Data files not found: {data_file}, {labels_file}\n"
            "Run scripts/download_data.py first."
        )

    epochs = np.load(data_file)   # (n_epochs, n_channels, n_samples)
    labels = np.load(labels_file)
    return epochs, labels


def main():
    parser = argparse.ArgumentParser(description="Train deep learning EEG classifiers")
    parser.add_argument("--task",    type=str, default="focus",    choices=["focus", "stress", "sleep"])
    parser.add_argument("--model",   type=str, default="cnn",      choices=["cnn", "lstm", "cnn_lstm"])
    parser.add_argument("--config",  type=str, default=None)
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience",   type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="models/deep")
    args = parser.parse_args()

    config_path = args.config or f"configs/{args.task}_config.yaml"
    cfg = load_config(config_path) if os.path.exists(config_path) else {}

    print(f"\n{'='*60}")
    print(f"Task: {args.task.upper()}  |  Model: {args.model.upper()}")
    print(f"{'='*60}")

    # Data
    epochs, labels = load_epochs(cfg, args.task)
    n_classes = len(np.unique(labels))
    class_names = TASK_LABELS.get(args.task, [str(i) for i in range(n_classes)])
    input_shape = epochs.shape[1:]   # (n_channels, n_samples)

    print(f"Epochs: {epochs.shape}  |  Classes: {n_classes}  |  Input shape: {input_shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        epochs, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Build model
    build_fn = MODEL_BUILDERS[args.model]
    keras_model = build_fn(input_shape=input_shape, n_classes=n_classes)
    keras_model.summary()

    task_type = "binary" if n_classes == 2 else "multiclass"
    clf = EEGDeepClassifier(keras_model, n_classes=n_classes, task=task_type)

    # Train
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, f"{args.task}_{args.model}_best.keras")

    history = clf.train(
        X_train, y_train,
        X_val=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        save_best_path=best_model_path,
    )

    # Evaluate
    results = clf.evaluate(X_test, y_test)

    # Plots
    plot_training_history(
        history,
        save_path=os.path.join(args.output_dir, f"{args.task}_{args.model}_training.png"),
    )
    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names=class_names,
        title=f"{args.model.upper()} — {args.task.capitalize()} Confusion Matrix",
        save_path=os.path.join(args.output_dir, f"{args.task}_{args.model}_cm.png"),
    )

    # ROC curves for multiclass
    if n_classes > 2:
        y_prob = keras_model.predict(X_test, verbose=0)
        plot_roc_curves(
            y_test, y_prob, class_names,
            save_path=os.path.join(args.output_dir, f"{args.task}_{args.model}_roc.png"),
        )

    print(f"\nFinal test accuracy: {results['accuracy']:.4f}")
    print(f"Final test F1 (macro): {results['f1_macro']:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
