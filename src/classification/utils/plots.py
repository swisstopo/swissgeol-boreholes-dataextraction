"""Plotting utilities for model evaluation."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from classification.utils.datasets.classification import ClassificationSystem


def plot_confusion_matrix(
    cm: np.ndarray,
    out_directory: Path,
    split: str = "test",
    all_classes: list[ClassificationSystem.EnumMember] | None = None,
) -> tuple[Path, Path]:
    """Generate and save a confusion matrix as CSV (all classes) and PNG (active classes only).

    Args:
        cm: Square confusion matrix of shape (n_classes, n_classes), row = true label, col = predicted.
        out_directory: Directory to save the outputs.
        split: Dataset split name used in filenames and the figure title.
        all_classes: Ordered list of all classes. Must match the row/column order of cm.

    Returns:
        Paths to the saved CSV and PNG respectively.
    """
    classes = sorted(all_classes, key=lambda c: c.value)
    class_names = [cls.name for cls in classes]

    out_directory.mkdir(parents=True, exist_ok=True)

    csv_path = out_directory / f"confusion_matrix_{split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, cm, strict=True):
            writer.writerow([name] + row.tolist())

    # Filter to classes with at least one true or predicted sample for a readable plot.
    active = (cm.sum(axis=1) > 0) | (cm.sum(axis=0) > 0)
    active_idx = np.where(active)[0]
    cm_plot = cm[np.ix_(active_idx, active_idx)]
    names_plot = [class_names[i] for i in active_idx]

    png_path = out_directory / f"confusion_matrix_{split}.png"

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_plot, display_labels=names_plot)
    disp.plot(cmap="Blues", xticks_rotation=45)
    disp.ax_.set_title(f"Confusion Matrix — {split}")
    fig = disp.figure_

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return csv_path, png_path
