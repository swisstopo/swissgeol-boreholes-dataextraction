"""Dataset slice registry and configuration for color classification experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from classification import DATAPATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slice registry
# Each entry maps a short name to (ground_truth_path, consolidated).
#   consolidated = 1    → consolidated layers only
#   consolidated = 0    → unconsolidated layers only
#   consolidated = None → all layers regardless of consolidation type
# ---------------------------------------------------------------------------
DATASET_REGISTRY: dict[str, tuple[Path, int | None]] = {
    "ZH-conso": (DATAPATH / "zurich_ground_truth.json", 1),
    "ZH-unconso": (DATAPATH / "zurich_ground_truth.json", 0),
    "GQ-conso": (DATAPATH / "geoquat_ground_truth.json", 1),
    "GQ-unconso": (DATAPATH / "geoquat_ground_truth.json", 0),
    "TH-conso": (DATAPATH / "thurgau_ground_truth.462.json", 1),
    "TH-unconso": (DATAPATH / "thurgau_ground_truth.462.json", 0),
    "DW-conso": (DATAPATH / "deepwells_ground_truth.json", 1),
    "DW-unconso": (DATAPATH / "deepwells_ground_truth.json", 0),
    "NA-conso": (DATAPATH / "nagra_ground_truth.json", 1),
    "NA-unconso": (DATAPATH / "nagra_ground_truth.json", 0),
}


@dataclass(frozen=True)
class ColorDatasetConfig:
    """Configuration for a color classification train/test experiment.

    Attributes:
        train_slices: Slice names (from DATASET_REGISTRY) used for training.
        test_slices:  Slice names (from DATASET_REGISTRY) used for evaluation.
        name:         Optional experiment name used for logging and output directories.
    """

    train_slices: list[str]
    test_slices: list[str]
    name: str = "color_experiment"

    def __post_init__(self) -> None:
        unknown = [s for s in self.train_slices + self.test_slices if s not in DATASET_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown dataset slices: {unknown}. Available: {list(DATASET_REGISTRY)}")


ALL_CONSO_SLICES: list[str] = ["ZH-conso", "GQ-conso", "TH-conso", "DW-conso", "NA-conso"]
ALL_UNCONSO_SLICES: list[str] = ["ZH-unconso", "GQ-unconso", "TH-unconso", "DW-unconso", "NA-unconso"]
