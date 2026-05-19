"""Dataset slice registry and configuration for color classification experiments.

Datasets are split into conso/unconso slices so we can evaluate whether
color labels generalise across consolidation types.

Usage:
    config = ColorDatasetConfig(
        train_slices=["ZH-conso"],
        test_slices=["ZH-unconso", "NA-unconso"],
    )
    train_ds, val_ds, test_ds = load_splits(config)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from classification import DATAPATH
from classification.data_loader.color_data_loader import BoreholeDataset

logger = logging.getLogger(__name__)

# Predictions file that provides material descriptions for Thurgau layers.
# The ground truth for Thurgau has color labels but no material_description fields;
# this predictions file (produced with perfect-depth alignment) fills that gap.
_THURGAU_PRED_PATH = DATAPATH / "output/thurgau_perfectdepth/predictions.json"

# ---------------------------------------------------------------------------
# Slice registry
# Each entry maps a short name to (ground_truth_path, consolidated) or
# (ground_truth_path, consolidated, pred_path) when a predictions file is
# needed to supplement missing material descriptions in the ground truth.
#   consolidated = 1    → consolidated layers only
#   consolidated = 0    → unconsolidated layers only
#   consolidated = None → all layers regardless of consolidation type
# ---------------------------------------------------------------------------
DATASET_REGISTRY: dict[str, tuple[Path, int | None] | tuple[Path, int | None, Path]] = {
    "ZH-conso": (DATAPATH / "zurich_ground_truth.json", 1),
    "ZH-unconso": (DATAPATH / "zurich_ground_truth.json", 0),
    "GQ-conso": (DATAPATH / "geoquat_ground_truth.json", 1),
    "GQ-unconso": (DATAPATH / "geoquat_ground_truth.json", 0),
    "TH-conso": (DATAPATH / "thurgau_ground_truth.json", 1, _THURGAU_PRED_PATH),
    "TH-unconso": (DATAPATH / "thurgau_ground_truth.json", 0, _THURGAU_PRED_PATH),
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


def consolidation_study_configs() -> list[ColorDatasetConfig]:
    """Return the 6 canonical configs for the consolidation generalisation study.

    The 6 experiments answer the question: does training on consolidated /
    unconsolidated / mixed samples affect color detection on each split?

        conso   → conso    (in-distribution baseline)
        conso   → unconso  (cross-type generalisation)
        unconso → conso    (cross-type generalisation)
        unconso → unconso  (in-distribution baseline)
        mixed   → conso    (does adding unconso training data help on conso test?)
        mixed   → unconso  (does adding conso training data help on unconso test?)
    """
    return [
        ColorDatasetConfig(ALL_CONSO_SLICES, ALL_CONSO_SLICES, "conso_to_conso"),
        ColorDatasetConfig(ALL_CONSO_SLICES, ALL_UNCONSO_SLICES, "conso_to_unconso"),
        ColorDatasetConfig(ALL_UNCONSO_SLICES, ALL_CONSO_SLICES, "unconso_to_conso"),
        ColorDatasetConfig(ALL_UNCONSO_SLICES, ALL_UNCONSO_SLICES, "unconso_to_unconso"),
        ColorDatasetConfig(ALL_CONSO_SLICES + ALL_UNCONSO_SLICES, ALL_CONSO_SLICES, "mixed_to_conso"),
        ColorDatasetConfig(ALL_CONSO_SLICES + ALL_UNCONSO_SLICES, ALL_UNCONSO_SLICES, "mixed_to_unconso"),
    ]


def _load_pred_lookup(pred_path: Path) -> dict[str, dict[tuple, str]]:
    """Build a material-description lookup from a predictions JSON.

    Parses the predictions format produced by the extraction pipeline and returns a nested
    mapping that from_json uses to fill in material descriptions absent from the ground truth.

    Args:
        pred_path: Path to the predictions JSON
            (maps filename → {boreholes: [{borehole_index, layers: [{material_description, depths}]}]}).

    Returns:
        Mapping of filename → {(borehole_index, depth_start, depth_end) → description_text}.
        Files or layers with no usable text are omitted.
    """
    if not pred_path.exists():
        logger.warning(
            "Predictions file not found: %s — layers without a ground-truth material_description will be skipped.",
            pred_path,
        )
        return {}

    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    lookup: dict[str, dict[tuple, str]] = {}
    for filename, file_data in predictions.items():
        file_lookup: dict[tuple, str] = {}
        for bh in file_data["boreholes"]:
            bi = bh.get("borehole_index")
            for layer in bh["layers"]:
                depths = layer.get("depths") or {}
                start = (depths.get("start") or {}).get("value")
                end = (depths.get("end") or {}).get("value")
                text = (layer.get("material_description") or {}).get("text")
                if text:
                    file_lookup[(bi, start, end)] = text
        if file_lookup:
            lookup[filename] = file_lookup

    logger.info("Loaded prediction descriptions for %d files from %s.", len(lookup), pred_path)
    return lookup


def load_splits(config: ColorDatasetConfig) -> tuple[BoreholeDataset, BoreholeDataset, BoreholeDataset]:
    """Resolve a ColorDatasetConfig into train, val, and test BoreholeDatasets.

    For slices whose registry entry includes a predictions path (e.g. TH-conso), material
    descriptions are supplemented from that predictions file before splitting.

    Args:
        config: The dataset configuration to load.

    Returns:
        A (train_dataset, val_dataset, test_dataset) tuple.
    """
    pred_lookup: dict[str, dict[tuple, str]] = {}
    for name in set(config.train_slices + config.test_slices):
        entry = DATASET_REGISTRY[name]
        if len(entry) == 3:
            pred_lookup.update(_load_pred_lookup(entry[2]))

    train_slices = [
        (path, consolidated) for path, consolidated, *_ in (DATASET_REGISTRY[name] for name in config.train_slices)
    ]
    test_slices = [
        (path, consolidated) for path, consolidated, *_ in (DATASET_REGISTRY[name] for name in config.test_slices)
    ]
    return BoreholeDataset.from_splits(train=train_slices, test=test_slices, pred_lookup=pred_lookup or None)
