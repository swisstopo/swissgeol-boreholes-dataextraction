"""Borehole type classification dataset module."""

from __future__ import annotations

from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem


class BoreholeTypeSystem(ClassificationSystem):
    """Classification system for the type of a borehole (e.g. borehole, trial pit, penetration test).

    The ground truth for this system lives on the borehole's metadata.
    """

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a borehole type class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "trial pit").

        Returns:
            str: The normalized borehole type class string (e.g., "trial_pit").
        """
        return class_str.lower().replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[BoreholeTypeClasses]:
        """Return the BoreholeTypeClasses Enum."""
        return cls.BoreholeTypeClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "borehole_type"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[list[str]]:
        """Return a list of keys that retrieve the ground truth class string from the borehole's metadata."""
        return [["metadata", "borehole_type"]]

    @classmethod
    def is_document_level(cls) -> bool:
        """Train on one example per borehole (full/header text of the borehole), not per layer."""
        return True

    @classmethod
    def get_default_class_value(cls) -> BoreholeTypeClasses:
        """Return the default value for the enum class."""
        return cls.BoreholeTypeClasses.not_specified

    class BoreholeTypeClasses(IntEnum):
        """Borehole type classes (0-based indexing)."""

        not_specified = 0
        borehole = auto()
        other = auto()
        penetration_test = auto()
        trial_pit = auto()
