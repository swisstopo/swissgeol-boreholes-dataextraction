"""Cementation classification dataset module."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem

logger = logging.getLogger(__name__)


class CementationSystem(ClassificationSystem):
    """Classification system for cementation of consolidated geological layers."""

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a cementation class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "weakly cemented").

        Returns:
            str: The normalized cementation class string (e.g., "weakly_cemented").
        """
        return class_str.lower().replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[CementationClasses]:
        """Return the CementationClasses Enum."""
        return cls.CementationClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "cementation"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["consolidated", "cementation"]

    @classmethod
    def get_default_class_value(cls) -> CementationClasses:
        """Return the default value for the enum class."""
        return cls.CementationClasses.not_specified

    class CementationClasses(IntEnum):
        """Cementation classes for consolidated soil classification (0-based indexing)."""

        not_specified = 0
        moderately_cemented = auto()
        other = auto()
        strongly_cemented = auto()
        uncemented = auto()
        weakly_cemented = auto()
        well_cemented = auto()
