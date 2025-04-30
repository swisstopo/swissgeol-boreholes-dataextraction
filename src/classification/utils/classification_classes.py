"""Class module."""

import logging
from enum import IntEnum, auto

logger = logging.getLogger(__name__)


class USCSClasses(IntEnum):
    """USCS (Unified Soil Classification System) classes.

    Note:
        By default, auto() assigns integer values starting from 1. In Python, especially in machine learning, it is
        common to start class labels from 0. The Trainer used when training the BERT model expects labels starting at
        0, so using 0-based indexing here avoids the need to address the issue later and prevents potential bugs.
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


class LithologyClasses(IntEnum):
    """Material classes for consolidated soil classification.

    Those classes are selected from the swissgeol-lexic-vocabulary-lithologie repository. Only the main classes are
    selected, meaning that their name does not contain a ':' and none of their narrower class is also a main class.

    We should try to reduce the number of classes (that should only be 15-20).

    Note:
        0-based indexing is used to maintain consistency with machine learning labeling conventions.
    """

    kA = 0
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
    Clay = auto()
    Loam = auto()  # not seen in ground truth data (yet) ..
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
    Marl = auto()  # not seen in ground truth data (yet)
    Marlstone = auto()
    Pebble = auto()  # not seen in ground truth data (yet)
    Sand = auto()
    Silt = auto()  # not seen in ground truth data (yet)
    Peat = auto()  # not seen in ground truth data (yet)
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


# Define joined type to use everywere in the code
ClassEnum = USCSClasses | LithologyClasses


def map_most_similar_class(class_str: str, data_type: str) -> ClassEnum:
    """Maps a given string to the closest matching class in the provided data_type.

    This function normalizes the input string depending on the data type (uscs or lithology) and tries to find a
    matching class name. If no match is found, it returns the default class `kA`.

    Args:
        class_str (str): The input string to map.
        data_type (str): The data_type to search in (uscs or lithology).

    Returns:
        ClassEnum: The matching enum member, or `kA` if no match is found.
    """
    class_str = class_str.lower()
    normalized_str = class_str.replace("-", "_") if data_type == "uscs" else class_str.split(",")[0]

    classes_enum = USCSClasses if data_type == "uscs" else LithologyClasses
    for class_ in classes_enum:
        if normalized_str == class_.name.lower():
            return class_
    logger.warning(f"{class_str} does not have a matching class, mapping it to {classes_enum.kA.name} instead.")
    return classes_enum.kA  # keine Angabe = no indication
