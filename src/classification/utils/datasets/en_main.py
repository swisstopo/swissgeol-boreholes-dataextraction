"""ENMain classification dataset module."""

from __future__ import annotations

import keyword
import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem, ClassificationTask

logger = logging.getLogger(__name__)


class ENMainSystem(ClassificationSystem):
    """Implementation of the main-level EN classification system.

    This class provides the EN classes at the main level.
    """

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a EN class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "Or", "Gr").

        Returns:
            str: The normalized EN class string. Python keywords get a trailing underscore
                to match enum member names (e.g. "Or" → "or_", "Gr" → "gr").
        """
        normalized = class_str.lower()
        return normalized + "_" if keyword.iskeyword(normalized) else normalized

    @classmethod
    def get_enum(cls) -> type[ENMainClasses]:
        """Return the ENClasses Enum."""
        return cls.ENMainClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system."""
        return "en_main"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[list[str]]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return [["unconsolidated", "main"]]

    @classmethod
    def get_default_class_value(cls) -> ENMainClasses:
        """Default value for the enum (not specified)."""
        return cls.ENMainClasses.ns

    class ENMainClasses(IntEnum):
        """Complete EN main class list (0-based indexing)."""

        ns = 0  # not specified
        an = auto()  # anthropogenic soil
        ba = auto()  # backfill
        bo = auto()  # boulder
        cgr = auto()  # coarse gravel
        cl = auto()  # clay
        co = auto()  # cobbles
        csa = auto()  # coarse sand
        fgr = auto()  # fine gravel
        fmgr = auto()  # fine-medium gravel
        fmsa = auto()  # fine-medium sand
        fsa = auto()  # fine sand
        gr = auto()  # gravel
        hu = auto()  # humus
        lbo = auto()  # large boulder
        mcgr = auto()  # medium-coarse gravel
        mcsa = auto()  # medium-coarse sand
        mgr = auto()  # medium gravel
        msa = auto()  # medium sand
        or_ = auto()  # organic soil
        oth = auto()  # other
        pt = auto()  # peat
        sa = auto()  # sand
        si = auto()  # silt


class ENSecondarySystem(ENMainSystem):
    """Multi-label EN classification system combining main and secondary unconsolidated classes."""

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system."""
        return "en_secondary"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[list[str]]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return [["unconsolidated", "main"], ["unconsolidated", "other"]]

    @classmethod
    def classification_task(cls) -> ClassificationTask:
        """Return the multi-label classification task type for this system."""
        return ClassificationTask.rank
