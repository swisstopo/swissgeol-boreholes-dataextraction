"""Organic component classification system for unconsolidated soils (organic_components field)."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem, ClassificationTask

logger = logging.getLogger(__name__)


class OrganicComponentsSystem(ClassificationSystem):
    """Classification system for the organic components of unconsolidated geological layers."""

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize an organic components class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "remains of wood").

        Returns:
            str: The normalized organic components class string (e.g., "remains_of_wood").
        """
        return class_str.lower().replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[OrganicComponentsClasses]:
        """Return the OrganicComponentsClasses Enum."""
        return cls.OrganicComponentsClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system."""
        return "organic_components"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["unconsolidated", "organic_components"]

    @classmethod
    def get_default_class_value(cls) -> OrganicComponentsClasses:
        """Return the default value for the enum class."""
        return cls.OrganicComponentsClasses.not_specified

    @classmethod
    def classification_task(cls) -> ClassificationTask:
        return ClassificationTask.multi_label

    class OrganicComponentsClasses(IntEnum):
        """Complete organic components class list (0-based indexing)."""

        not_specified = 0
        coal = auto()
        earth = auto()
        humus = auto()
        other = auto()
        peat = auto()
        remains_of_plants = auto()
        remains_of_wood = auto()
        roots = auto()
        undifferenciated_organic_material = auto()
        varves = auto()
