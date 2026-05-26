"""TODO."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem

logger = logging.getLogger(__name__)


class ENMainSystem(ClassificationSystem):
    """Implementation of the main-level EN classification system.

    This class provides the EN classes at the main level.
    """

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a EN class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "Or").

        Returns:
            str: The normalized EN class string (e.g., "or").
        """
        return class_str.lower()

    @classmethod
    def get_enum(cls) -> type[ENMainClasses]:
        """Return the ENClasses Enum."""
        return cls.ENMainClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system."""
        return "EN_main"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["unconsolidated", "main"]

    @classmethod
    def get_default_class_value(cls) -> ENMainClasses:
        """Default value for the enum (not specified)."""
        return cls.ENMainClasses.ns

    @classmethod
    def get_dummy_classifier_class_value(cls) -> ENMainClasses:
        """Return a dummy value."""
        return cls.ENMainClasses.lbo

    class ENMainClasses(IntEnum):
        """Complete EN main class list (0-based indexing)."""

        lbo = 0  # large boulder
        bo = auto()  # boulder
        co = auto()  # cobbles
        gr = auto()  # gravel
        cgr = auto()  # coarse gravel
        mcgr = auto()  # medium-coarse gravel
        mgr = auto()  # medium gravel
        fmgr = auto()  # fine-medium gravel
        fgr = auto()  # fine gravel
        sa = auto()  # sand
        csa = auto()  # coarse sand
        mcsa = auto()  # medium-coarse sand
        msa = auto()  # medium sand
        fmsa = auto()  # fine-medium sand
        fsa = auto()  # fine sand
        si = auto()  # silt
        cl = auto()  # clay
        pt = auto()  # peat
        or_ = auto()  # organic soil
        hu = auto()  # humus
        an = auto()  # anthropogenic soil
        ba = auto()  # backfill
        oth = auto()  # other
        ns = auto()  # not specified
