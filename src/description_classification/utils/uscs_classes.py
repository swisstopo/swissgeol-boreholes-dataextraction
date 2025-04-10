"""class module."""

import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class USCSClasses(Enum):
    """USCS (Unified Soil Classification System) classes."""

    def _generate_next_value_(name, start, count, last_values):
        return count  # To start from 0 instead of 1

    kunst = auto()
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


def map_most_similar_uscs(uscs_str: str) -> USCSClasses:
    """Maps the ground truth string to one of the USCSClasses.

    Args:
        uscs_str (str): the ground truth string

    Returns:
        USCSClasses: the matching class
    """
    for class_ in USCSClasses:
        if uscs_str.replace("-", "_") == class_.name:
            return class_
    logger.warning(f"{uscs_str} does have a matching class, mapping it to {USCSClasses.kA.name} instead.")
    return USCSClasses.kA  # keine Angabe = no indication
