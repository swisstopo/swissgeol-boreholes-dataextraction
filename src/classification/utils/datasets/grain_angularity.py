"""Grain angularity classification dataset module."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem, ClassificationTask

logger = logging.getLogger(__name__)


class GrainAngularitySystem(ClassificationSystem):
    """Classification system for grain angularity of unconsolidated geological layers."""

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a grain angularity class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "sub-angular").

        Returns:
            str: The normalized grain angularity class string (e.g., "sub_angular").
        """
        return class_str.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[GrainAngularityClasses]:
        """Return the GrainAngularityClasses Enum."""
        return cls.GrainAngularityClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "grain_angularity"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[list[str]]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return [["unconsolidated", "grain_angularity"]]

    @classmethod
    def get_default_class_value(cls) -> GrainAngularityClasses:
        """Return the default value for the enum class."""
        return cls.GrainAngularityClasses.not_specified

    @classmethod
    def classification_task(cls) -> ClassificationTask:
        return ClassificationTask.multi_label

    class GrainAngularityClasses(IntEnum):
        """Grain angularity classes for unconsolidated soil classification (0-based indexing)."""

        not_specified = 0
        angular = auto()
        other = auto()
        rounded = auto()
        sub_angular = auto()
        sub_rounded = auto()
        very_angular = auto()
        well_rounded = auto()
