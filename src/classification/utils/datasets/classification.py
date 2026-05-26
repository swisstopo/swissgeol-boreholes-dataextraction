"""Classification dataset module."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from functools import reduce

from core.ground_truth import GroundTruth, GroundTruthLayer

logger = logging.getLogger(__name__)


@dataclass
class LayerInformation:
    """Class for each layer in the ground truth json file.

    A layer is either classified into USCS or lithology, but never both.
    """

    filename: str
    borehole_index: int
    layer_index: int
    language: str
    material_description: str
    class_system: type[ClassificationSystem]
    ground_truth_class: None | ClassificationSystem.EnumMember
    prediction_class: None | ClassificationSystem.EnumMember
    llm_reasoning: None | str


class ClassificationSystem(ABC):
    """Abstract base class for classification system.

    This class defines the core structure and methods that all classification systems
    should implement. It defines methods for normalizing input class strings, returning
    the corresponding Enum class, and providing a default value for dummy classification.
    """

    EnumClassType = type[IntEnum]  # Type alias for the class that inherit InEnum (e.g. USCSClasses)
    EnumMember = IntEnum  # Type alias for a member of those class (e.g. USCSClasses.CL_ML)

    @classmethod
    @abstractmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize input class string."""
        ...

    @classmethod
    @abstractmethod
    def get_enum(cls) -> EnumClassType:
        """Return the Enum type associated with the classification."""
        ...

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        ...

    @classmethod
    @abstractmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        ...

    @classmethod
    @abstractmethod
    def process_layer(
        cls,
        filename: str,
        borehole_index: int,
        layer_index: int,
        layer: GroundTruthLayer,
    ) -> LayerInformation | None:
        """Convert a single ground truth layer into a LayerInformation entry if possible, or return None."""
        ...

    @classmethod
    def reduce_label(
        cls,
        layer: GroundTruthLayer,
    ) -> int | None:
        """Extract the integer class index from a layer by resolving the ground truth key path, or None if absent."""
        try:
            label_str = reduce(getattr, cls.get_layer_ground_truth_keys(), layer)
        except AttributeError:
            return None

        if label_str is None:
            return None

        return cls.map_most_similar_class(label_str)

    @classmethod
    def process(cls, gt: GroundTruth) -> list[LayerInformation]:
        """Extract all labelled layers from a GroundTruth object as a flat list of LayerInformation entries.."""
        return [
            LayerInformation(
                filename=filename,
                borehole_index=borehole_index,
                layer_index=layer_index,
                language="",
                material_description=layer.material_description,
                class_system=cls,
                ground_truth_class=cls.reduce_label(layer),
                prediction_class=None,
                llm_reasoning=None,
            )
            for filename, boreholes in gt.ground_truth.items()
            for borehole_index, borehole in enumerate(boreholes)
            for layer_index, layer in enumerate(borehole.layers)
            if cls.reduce_label(layer) is not None and layer.material_description is not None
        ]

    @classmethod
    def get_class_from_entry(cls, entry: dict, keys: list[str]) -> str | None:
        """Returns the class of the classification system used from a possibly nested entry.

        If one of the entries is missing from the nested structure, returns None.
        """
        return (
            cls.get_class_from_entry(entry=entry.get(keys[0]), keys=keys[1:])
            if keys and isinstance(entry, dict)
            else entry
        )

    @classmethod
    @abstractmethod
    def get_dummy_classifier_class_value(cls) -> EnumMember:
        """Return a default value for dummy classification."""
        ...

    @classmethod
    @abstractmethod
    def get_default_class_value(cls) -> EnumMember:
        """Return the default value for the enum class."""
        ...

    @classmethod
    def map_most_similar_class(cls, class_str: str) -> EnumMember:
        """Maps a given string to the closest matching class in the classification system.

        This function normalizes the input string depending on the data type (uscs or lithology) and tries to find a
        matching class name. If no match is found, it returns the default class `kA`.

        Args:
            class_str (str): The input string to map.

        Returns:
            ClassificationType.EnumMember: The matching enum member, or default if no match is found.
        """
        normalized_str = cls.normalize_class_string(class_str)

        classes_enum = cls.get_enum()
        for class_ in classes_enum:
            if normalized_str == class_.name.lower():
                return class_
        logger.warning(
            f"{class_str} does not have a matching class, mapping it to {cls.get_default_class_value().name} instead."
        )
        return cls.get_default_class_value()
