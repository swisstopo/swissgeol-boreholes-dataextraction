"""USCS classification dataset module."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem

logger = logging.getLogger(__name__)


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
        return class_str[0].lower().replace("-", "_")

    @classmethod
    def get_enum(cls) -> type[USCSClasses]:
        """Return the USCSClasses Enum."""
        return cls.USCSClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "uscs"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["unconsolidated", "uscs"]

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
