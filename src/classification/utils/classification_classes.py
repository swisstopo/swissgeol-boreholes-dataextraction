"""Class module."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum, IntEnum, auto
from typing import Literal

logger = logging.getLogger(__name__)


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
    def get_layer_ground_truth_key(cls) -> str:
        """Return the key in the layer dictionary that retrieves the ground truth class string."""
        ...

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
            ClassificationType.EnumMember: The matching enum member, or `kA` if no match is found.
        """
        normalized_str = cls.normalize_class_string(class_str)

        classes_enum = cls.get_enum()
        for class_ in classes_enum:
            if normalized_str == class_.name.lower():
                return class_
        logger.warning(f"{class_str} does not have a matching class, mapping it to {classes_enum.kA.name} instead.")
        return cls.get_default_class_value()


class USCSSystem(ClassificationSystem):
    """Implementation of a classification type based on the USCS (Unified Soil Classification System).

    This class implements the methods defined in `ClassificationType` for the USCS classification system,
    which is commonly used to classify unconsolidated soils.
    """

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a USCS class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "CL-ML").

        Returns:
            str: The normalized USCS class string (e.g., "cl_ml").

        """
        return class_str.lower().replace("-", "_")

    @classmethod
    def get_enum(cls) -> type[USCSClasses]:
        """Return the USCSClasses Enum."""
        return cls.USCSClasses

    @classmethod
    def get_layer_ground_truth_key(cls) -> str:
        """Return the key in the layer dictionary that retrieves the ground truth class string."""
        return "uscs_1"

    @classmethod
    def get_default_class_value(cls) -> USCSClasses:
        """Return the default value for the enum class."""
        return cls.USCSClasses.kA  # keine Angabe = no indication

    @classmethod
    def get_dummy_classifier_class_value(cls) -> USCSClasses:
        """Return the default value CL_ML for the dummy classifier."""
        return cls.USCSClasses.CL_ML

    class USCSClasses(IntEnum):
        """USCS (Unified Soil Classification System) classes.

        Note:
            By default, auto() assigns integer values starting from 1. In Python, especially in machine learning, it is
            common to start class labels from 0. The Trainer used when training the BERT model expects labels starting
            at 0, so using 0-based indexing avoids the need to address the issue later and prevents potential bugs.
        """

        kunst = 0
        Bl = auto()
        GP = auto()
        CH = auto()
        CM = auto()
        CL = auto()
        CL_ML = auto()
        G = auto()
        S = auto()
        GW_GC = auto()
        Pt = auto()
        ML = auto()
        GM = auto()
        kA = auto()
        FELS = auto()
        SC = auto()
        S_SM = auto()
        SM = auto()
        SP = auto()
        SP_SC = auto()
        SP_SM = auto()
        SW = auto()
        SW_SC = auto()
        SW_SM = auto()
        G_GC = auto()
        G_GM = auto()
        St = auto()
        St_Bl = auto()
        OH = auto()
        OL = auto()
        S_SC = auto()
        SC_SM = auto()
        GC = auto()
        GC_GM = auto()
        GP_GC = auto()
        GP_GM = auto()
        GW = auto()
        GW_GM = auto()
        MH = auto()


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
    def get_layer_ground_truth_key(cls) -> str:
        """Return the key in the layer dictionary that retrieves the ground truth class string."""
        return "lithology"

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


class ExistingClassificationSystems(Enum):
    """Enum listing all existing classification types.

    The value of each entry is the Classification system class, not an instance of the class.
    """

    uscs = USCSSystem
    lithology = LithologySystem

    @classmethod
    def get_classification_system_type(cls, class_system: Literal["uscs", "lithology"]) -> type[ClassificationSystem]:
        """Returns the class of a classification system based on input string.

        Args:
            class_system (Literal["uscs", "lithology"]): The name of the classification system.

        Returns:
            Type[ClassificationSystem]: The associated ClassificationSystem class.

        """
        return cls[class_system].value
