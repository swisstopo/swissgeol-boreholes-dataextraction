"""Dataset slice registry and configuration for color classification experiments."""

from __future__ import annotations

import logging
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
