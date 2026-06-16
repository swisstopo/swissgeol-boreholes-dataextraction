"""Lithology classification dataset module."""

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

    UNCONSOLIDATED_KEYWORDS = ["clay", "marl", "silt", "peat", "sand", "pebble", "loam", "unconsolidated"]

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a lithology class string.

        Examples include texts that are semicolon/comma separated (e.g. "limestone: micritic" -> "limestone",
        "sandstone, marly" -> "sandstone") or white space (e.g. "not specified" -> "not_specified")

        Args:
            class_str (str): The class string to be normalized.

        Returns:
            str: The normalized lithology class string.

        """
        class_str = class_str.lower()
        class_str = class_str.split(":")[0].strip()
        class_str = class_str.split(",")[0].strip()
        class_str = class_str.replace(" ", "_")
        return class_str

    @classmethod
    def get_enum(cls) -> type[LithologyClasses]:
        """Return the LithologyClasses Enum."""
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
        return cls.LithologyClasses.not_specified

    @classmethod
    def map_most_similar_class(cls, class_str: str) -> ClassificationSystem.EnumMember:
        """Maps a given string to the closest matching class in the classification system.

        This function normalizes the input string depending on the data type (uscs or lithology) and tries to find a
        matching class name.  It first attempts standard class matching, then checks for unconsolidated soil
        keywords. If it finds no match, returns the default class.

        Args:
            class_str (str): The input string to map.

        Returns:
            EnumMember: The matching enum member, or default if no match is found.
        """
        normalized_str = cls.normalize_class_string(class_str)

        classes_enum = cls.get_enum()
        for class_ in classes_enum:
            if normalized_str == class_.name.lower():
                return class_

        # After the standard class match, check for unconsolidated soil
        if any(word in cls.UNCONSOLIDATED_KEYWORDS for word in normalized_str.split()):
            return cls.LithologyClasses.unconsolidated

        logger.warning(
            f"{class_str} does not have a matching class, mapping it to {classes_enum.not_specified.name} instead."
        )
        return cls.get_default_class_value()

    class LithologyClasses(IntEnum):
        """Material classes for consolidated soil classification.

        Those classes are selected from the swissgeol-lexic-vocabulary-lithologie repository. Only the main classes are
        selected, meaning that their name does not contain a ':' and none of their narrower class is also a main class.

        Note:
            0-based indexing is used to maintain consistency with machine learning labeling conventions.
        """

        not_specified = 0
        amphibolite = auto()
        andesite = auto()  # not seen in ground truth data (yet)
        aplite = auto()
        basalt = auto()  # not seen in ground truth data (yet)
        basanite = auto()  # not seen in ground truth data (yet)
        bentonite = auto()  # not seen in ground truth data (yet)
        breccia = auto()
        cataclasite = auto()
        claystone = auto()
        conglomerate = auto()
        dacite = auto()  # not seen in ground truth data (yet)
        diorite = auto()
        dolostone = auto()
        eclogite = auto()  # not seen in ground truth data (yet)
        evaporite = auto()
        foidite = auto()  # not seen in ground truth data (yet)
        foidolite = auto()  # not seen in ground truth data (yet)
        gabbro = auto()  # not seen in ground truth data (yet)
        gneiss = auto()
        granite = auto()
        granodiorite = auto()
        granofels = auto()  # not seen in ground truth data (yet)
        granophyre = auto()  # not seen in ground truth data (yet)
        granulite = auto()  # not seen in ground truth data (yet)
        hornfels = auto()  # not seen in ground truth data (yet)
        ignimbrite = auto()  # not seen in ground truth data (yet)
        kakirite = auto()
        latite = auto()  # not seen in ground truth data (yet)
        limestone = auto()
        marble = auto()
        marlstone = auto()
        migmatite = auto()
        monzonite = auto()  # not seen in ground truth data (yet) ..
        mylonite = auto()
        other = auto()
        pegmatite = auto()  # not seen in ground truth data (yet)
        pelite = auto()  # is parent of other primary
        peridotite = auto()  # not seen in ground truth data (yet)
        phyllite = auto()
        phonolite = auto()  # not seen in ground truth data (yet)
        prasinite = auto()  # not seen in ground truth data (yet)
        psephite = auto()  # is parent of other primary
        psammite = auto()  # is parent of other primary
        pseudotachyllite = auto()  # not seen in ground truth data (yet)
        pyroxenite = auto()  # not seen in ground truth data (yet)
        quartzite = auto()
        rauwacke = auto()  # not seen in ground truth data (yet)
        rhyolite = auto()
        rock = auto()  # is parent of other primary
        rodingite = auto()  # not seen in ground truth data (yet)
        sandstone = auto()
        schist = auto()
        serpentinite = auto()
        siltstone = auto()
        syenite = auto()  # not seen in ground truth data (yet)
        tephrite = auto()  # not seen in ground truth data (yet)
        tonalite = auto()  # not seen in ground truth data (yet)
        trachyte = auto()  # not seen in ground truth data (yet)
        tuff = auto()
        unconsolidated = auto()
