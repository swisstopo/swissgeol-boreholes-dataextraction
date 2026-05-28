"""cementation classification system for consolidated soils."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem

logger = logging.getLogger(__name__)


class CementationConsolidatedSystem(ClassificationSystem):
    """Classification system for the  cementation of consolidated geological layers."""

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a cementation class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "well cemented").

        Returns:
            str: The normalized cementation class string (e.g., "well_cemented").
        """
        return class_str.lower().replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[CementClasses]:
        """Return the CementClasses Enum."""
        return cls.CementClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system."""
        return "cementation"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["consolidated", "cementation"]

    @classmethod
    def get_default_class_value(cls) -> CementClasses:
        """Default value for the enum (not specified)."""
        return cls.CementClasses.not_specified

    class CementClasses(IntEnum):
        """Complete cementation class list (0-based indexing)."""

        text_cli_en = 0
        uncemented = auto()
        weakly_cemented = auto()
        moderately_cemented = auto()
        well_cemented = auto()
        strongly_cemented = auto()
        other = auto()
        not_specified = auto()
