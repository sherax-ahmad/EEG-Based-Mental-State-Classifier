"""
Evaluation utilities: metrics, confusion matrix plots, training curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import List, Optional, Dict
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a styled confusion matrix.

    Args:
        cm         : Confusion matrix from sklearn.metrics.confusion_matrix
        class_names: Label names for axes
        normalize  : Normalize by true class counts
        save_path  : If given, save figure to this path
    """
    if normalize:
        cm_disp = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = ".2f"
    else:
        cm_disp = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2), max(5, len(class_names))))
    sns.heatmap(
        cm_disp,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(
    history,
    metrics: List[str] = ("loss", "accuracy"),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation curves from a Keras History object.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history.history:
            ax.plot(history.history[metric], label=f"Train {metric}", linewidth=2)
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f"Val {metric}", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Training {metric.capitalize()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ROC curves (one-vs-rest for multiclass).

    Args:
        y_true     : (n_samples,) integer labels
        y_prob     : (n_samples, n_classes) predicted probabilities
        class_names: Label names
    """
    from sklearn.preprocessing import label_binarize
    n_classes = len(class_names)
    Y = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(Y[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_models(
    results: Dict[str, Dict],
    metric: str = "accuracy",
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing multiple model results.

    Args:
        results : {model_name: {metric: value, ...}}
        metric  : Metric key to compare
    """
    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, values, color=plt.cm.Paired.colors[: len(names)], width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Model Comparison — {metric}", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
