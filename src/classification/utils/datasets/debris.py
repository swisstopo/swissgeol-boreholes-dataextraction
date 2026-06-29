"""Unclassified coarse components (debris) classification system for unconsolidated soils (debris field)."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem, ClassificationTask

logger = logging.getLogger(__name__)


class DebrisSystem(ClassificationSystem):
    """Classification system for the unclassified coarse components (debris) of unconsolidated geological layers."""

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize an unclassified coarse components class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "erratic block").

        Returns:
            str: The normalized coarse components class string (e.g., "erratic_block").
        """
        return class_str.lower().replace(",", "").replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[DebrisClasses]:
        """Return the DebrisClasses Enum."""
        return cls.DebrisClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system."""
        return "debris"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[list[str]]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return [["unconsolidated", "debris"]]

    @classmethod
    def get_default_class_value(cls) -> DebrisClasses:
        """Return the default value for the enum class."""
        return cls.DebrisClasses.not_specified

    @classmethod
    def classification_task(cls) -> ClassificationTask:
        return ClassificationTask.multi_label

    class DebrisClasses(IntEnum):
        """Complete unclassifiable coarse components class list (0-based indexing)."""

        not_specified = 0
        bed_load = auto()
        erratic_block = auto()
        fragments_splitters = auto()
        other = auto()
        rubble = auto()
        tufa = auto()
