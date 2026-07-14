"""Alteration degree classification system for consolidated and unconsolidated soils."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem

logger = logging.getLogger(__name__)


class AlterationDegreeSystem(ClassificationSystem):
    """Base classification system for the alteration degree of geological layers.

    Use AlterationDegreeConsolidatedSystem or AlterationDegreeUnconsolidatedSystem — do not instantiate directly.
    """

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize an alteration degree class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "slightly weathered").

        Returns:
            str: The normalized alteration degree class string (e.g., "slightly_weathered").
        """
        return class_str.lower().replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[AlterationDegreeClasses]:
        """Return the AlterationDegreeClasses Enum."""
        return cls.AlterationDegreeClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system."""
        return "alteration_degree"

    @classmethod
    def get_default_class_value(cls) -> AlterationDegreeClasses:
        """Return the default value for the enum class."""
        return cls.AlterationDegreeClasses.not_specified

    class AlterationDegreeClasses(IntEnum):
        """Alteration degree class list (0-based indexing)."""

        not_specified = 0
        completely_weathered = auto()
        fresh = auto()
        highly_weathered = auto()
        moderately_weathered = auto()
        other = auto()
        slightly_weathered = auto()
        weathered = auto()


class AlterationDegreeConsolidatedSystem(AlterationDegreeSystem):
    """Classification system for the alteration degree of consolidated geological layers."""

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[list[str]]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return [["consolidated", "alteration_degree"]]

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "alteration_degree_consolidated"


class AlterationDegreeUnconsolidatedSystem(AlterationDegreeSystem):
    """Classification system for the alteration degree of unconsolidated geological layers."""

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[list[str]]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return [["unconsolidated", "alteration_degree"]]

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "alteration_degree_unconsolidated"
