"""Grain shape classification dataset module."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem

logger = logging.getLogger(__name__)


class GrainShapeSystem(ClassificationSystem):
    """Classification system for grain shape of unconsolidated geological layers."""

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a grain shape class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "platy").

        Returns:
            str: The normalized grain shape class string.
        """
        return class_str.lower().replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[GrainShapeClasses]:
        """Return the GrainShapeClasses Enum."""
        return cls.GrainShapeClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "grain_shape"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["unconsolidated", "grain_shape"]

    @classmethod
    def get_default_class_value(cls) -> GrainShapeClasses:
        """Return the default value for the enum class."""
        return cls.GrainShapeClasses.not_specified

    class GrainShapeClasses(IntEnum):
        """Grain shape classes for unconsolidated soil classification (0-based indexing)."""

        not_specified = 0
        cubic = auto()
        elongated = auto()
        other = auto()
        platy = auto()
