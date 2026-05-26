"""TODO."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem

logger = logging.getLogger(__name__)


class LithologySystem(ClassificationSystem):
    """Implementation of a classification type based on the Lithology classification system.

    This class implements the methods defined in `ClassificationType` for the Lithology classification system,
    which is commonly used to classify consolidated soils.
    """

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a lithology class string.

        Args:
            class_str (str): The class string to be normalized (e.g. " "limestone, bioclasts",").

        Returns:
            str: The normalized lithology class string (e.g., "limestone").

        """
        return class_str.lower().split(",")[0]

    @classmethod
    def get_enum(cls) -> type[LithologyClasses]:
        """Return the USCSClasses Enum."""
        return cls.LithologyClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "lithology"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["consolidated", "lithology"]

    @classmethod
    def get_default_class_value(cls) -> LithologyClasses:
        """Return the default value for the enum class."""
        return cls.LithologyClasses.kA  # keine Angabe = no indication

    @classmethod
    def get_dummy_classifier_class_value(cls) -> LithologyClasses:
        """Return a default dummy value."""
        return cls.LithologyClasses.Marlstone

    unconsolidated_keywords = ["clay", "marl", "silt", "peat", "sand", "pebble", "loam", "unconsolidated"]

    @classmethod
    def map_most_similar_class(cls, class_str: str) -> LithologyClasses:
        """Maps a given string to the closest matching class in the classification system.

        This function normalizes the input string depending on the data type (uscs or lithology) and tries to find a
        matching class name.  It first attempts standard class matching, then checks for unconsolidated soil
        keywords. If it finds no match, returns the default class `kA`.

        Args:
            class_str (str): The input string to map.

        Returns:
            ClassificationType.EnumMember: The matching enum member, or `kA` if no match is found.
        """
        normalized_str = cls.normalize_class_string(class_str)

        classes_enum = cls.get_enum()
        for class_ in classes_enum:
            if normalized_str == class_.name.lower():
                return class_

        # After the standard class match, check for unconsolidated soil
        if any(word in cls.unconsolidated_keywords for word in normalized_str.split()):
            return cls.LithologyClasses.Unconsolidated

        logger.warning(f"{class_str} does not have a matching class, mapping it to {classes_enum.kA.name} instead.")
        return cls.get_default_class_value()

    class LithologyClasses(IntEnum):
        """Material classes for consolidated soil classification.

        Those classes are selected from the swissgeol-lexic-vocabulary-lithologie repository. Only the main classes are
        selected, meaning that their name does not contain a ':' and none of their narrower class is also a main class.

        Note:
            0-based indexing is used to maintain consistency with machine learning labeling conventions.
        """

        kA = 0
        Unconsolidated = auto()
        Amphibolite = auto()
        Migmatite = auto()
        Andesite = auto()  # not seen in ground truth data (yet)
        Aplite = auto()
        Basalt = auto()  # not seen in ground truth data (yet)
        Basanite = auto()  # not seen in ground truth data (yet)
        Bentonite = auto()  # not seen in ground truth data (yet)
        Claystone = auto()
        Breccia = auto()
        Rock = auto()  # is parent of other primary
        Rhyolite = auto()
        Psephite = auto()  # is parent of other primary
        Tuffite = auto()  # not seen in ground truth data (yet) ..
        Cataclastite = auto()
        Pelite = auto()  # is parent of other primary
        Conglomerate = auto()
        Dacite = auto()  # not seen in ground truth data (yet)
        Diorite = auto()
        Monzonite = auto()  # not seen in ground truth data (yet) ..
        Dolostone = auto()
        Eclogite = auto()  # not seen in ground truth data (yet)
        Evaporite = auto()
        Foidite = auto()  # not seen in ground truth data (yet)
        Foidolite = auto()  # not seen in ground truth data (yet)
        Gabbro = auto()  # not seen in ground truth data (yet)
        Mylonite = auto()
        Gneiss = auto()
        Granite = auto()
        Granodiorite = auto()
        Granulite = auto()  # not seen in ground truth data (yet)
        Psammite = auto()  # is parent of other primary
        Schist = auto()
        Syenite = auto()  # not seen in ground truth data (yet)
        Granofels = auto()  # not seen in ground truth data (yet)
        Peridotite = auto()  # not seen in ground truth data (yet)
        Pyroxenite = auto()  # not seen in ground truth data (yet)
        Granophyre = auto()  # not seen in ground truth data (yet)
        Hornfels = auto()  # not seen in ground truth data (yet)
        Ignimbrite = auto()  # not seen in ground truth data (yet)
        Kakirite = auto()
        Latite = auto()  # not seen in ground truth data (yet)
        Limestone = auto()
        Marble = auto()
        Marlstone = auto()
        Phyllite = auto()
        Pegmatite = auto()  # not seen in ground truth data (yet)
        Siltstone = auto()
        Serpentinite = auto()
        Phonolite = auto()  # not seen in ground truth data (yet)
        Prasinite = auto()  # not seen in ground truth data (yet)
        Sandstone = auto()
        Pseudotachyllite = auto()  # not seen in ground truth data (yet)
        Quartzite = auto()
        Rauwacke = auto()  # not seen in ground truth data (yet)
        Rodingite = auto()  # not seen in ground truth data (yet)
        Tonalite = auto()  # not seen in ground truth data (yet)
        Tephrite = auto()  # not seen in ground truth data (yet)
        Trachyte = auto()  # not seen in ground truth data (yet)
