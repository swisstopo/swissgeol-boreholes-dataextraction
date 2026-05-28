"""Plotting utilities for model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from classification.utils.datasets.classification import ClassificationSystem


def _get_class_names(classes: list[ClassificationSystem.EnumMember]) -> list[str]:
    """Extract class names from a list of EnumMembers, sorted by value.

    Args:
        classes (list[ClassificationSystem.EnumMember]): List of class enum members.

    Returns:
        list[str]: Sorted class names.
    """
    return [cls.name for cls in sorted(classes, key=lambda c: c.value)]


def _compute_confusion_matrix(
    predictions: list[ClassificationSystem.EnumMember],
    labels: list[ClassificationSystem.EnumMember],
    all_classes: list[ClassificationSystem.EnumMember] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute confusion matrix and extract sorted class names.

    Args:
        predictions (list[ClassificationSystem.EnumMember]): Predicted classes.
        labels (list[ClassificationSystem.EnumMember]): Ground truth classes.
        all_classes (list[ClassificationSystem.EnumMember] | None): Full ordered class list.

    Returns:
        tuple[np.ndarray, list[str]]: Raw confusion matrix and class names.
    """
    if all_classes is None:
        all_classes = sorted(set(predictions) | set(labels), key=lambda c: c.value)
    class_names = _get_class_names(all_classes)

    # Map EnumMember -> integer index for sklearn
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    pred_idx = [class_to_idx[prediction] for prediction in predictions]
    label_idx = [class_to_idx[label] for label in labels]

    cm = confusion_matrix(label_idx, pred_idx, labels=list(range(len(all_classes))))
    return cm, class_names


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    fmt: str,
    out_path: Path,
    cmap: str = "Blues",
) -> None:
    """Render and save a single confusion matrix figure.

    Args:
        cm (np.ndarray): The confusion matrix to plot (may be raw or normalized).
        class_names (list[str]): Labels for each class, in row/column order.
        title (str): Figure title.
        fmt (str): Format string for cell annotations (e.g. 'd' for int, '.1%' for percent).
        out_path (Path): File path to save the figure to.
        cmap (str): Matplotlib colormap name.
    """
    n = len(class_names)
    fig_size = max(8, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=max(6, 10 - n // 5),
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    predictions: list[ClassificationSystem.EnumMember],
    labels: list[ClassificationSystem.EnumMember],
    out_directory: Path,
    split: str = "test",
    all_classes: list[ClassificationSystem.EnumMember] | None = None,
) -> tuple[Path, Path]:
    """Generate and save raw-count and percentage confusion matrix figures.

    Args:
        predictions (list[ClassificationSystem.EnumMember]): Predicted classes.
        labels (list[ClassificationSystem.EnumMember]): Ground truth classes.
        out_directory (Path): Directory to save the figures.
        split (str): Dataset split name used in filenames and titles (e.g. 'test', 'eval').
        all_classes (list[ClassificationSystem.EnumMember] | None): Full ordered class list.
            Pass this to ensure classes absent from predictions/labels still appear in the matrix.

    Returns:
        tuple[Path, Path]: Paths to the raw-count and percentage figures respectively.
    """
    cm, class_names = _compute_confusion_matrix(predictions, labels, all_classes)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    raw_path = out_directory / f"confusion_matrix_{split}_raw.png"
    pct_path = out_directory / f"confusion_matrix_{split}_pct.png"

    _plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title=f"Confusion Matrix — {split} (raw counts)",
        fmt="d",
        out_path=raw_path,
    )
    _plot_confusion_matrix(
        cm=cm_normalized,
        class_names=class_names,
        title=f"Confusion Matrix — {split} (% of true class)",
        fmt=".1%",
        out_path=pct_path,
    )

    return raw_path, pct_path
