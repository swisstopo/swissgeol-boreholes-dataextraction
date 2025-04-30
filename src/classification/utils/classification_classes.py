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

    ##### unique classes in deepwells #####
    # quartzite ##
    # breccia
    # conglomerate
    # marlstone
    # serpentinite ##
    # other #----
    # limestone
    # rhyolite
    # sandstone
    # diorite
    # clay
    # rock #!!!! not is Enum: parent of all
    # claystone
    # kakirite
    # granodiorite
    # pelite # !!!!not is Enum: parent of claystone, marlstone, siltstone
    # dolostone
    # phyllite
    # mylonite
    # aplite
    # psephite #!!!!!!! not is Enum:parent of conglomerate
    # schist
    # cataclastite
    # evaporite
    # amphibolite
    # siltstone
    # granite
    # not specified #!!!!not is Enum:!
    # unconsolidated deposits #!!!not is Enum:!
    # gneiss
    # marble

    ###### unique calsses in nagra #####
    # aplite
    # dolostone
    # migmatite ### not in deepwells
    # cataclastite
    # kakirite
    # sandstone
    # evaporite
    # clay
    # unconsolidated deposits #!!! not is Enum:
    # granite
    # breccia
    # conglomerate
    # siltstone
    # sand ### not in deepwells
    # pelite
    # claystone
    # limestone
    # gneiss
    # marlstone
    # rock

    kA = 0
    Amphibolite = auto()
    Migmatite = auto()
    Andesite = auto()  #
    Aplite = auto()
    Basalt = auto()  #
    Basanite = auto()  #
    Bentonite = auto()  #
    Claystone = auto()
    Breccia = auto()
    Rhyolite = auto()
    Tuffite = auto()  #
    Cataclastite = auto()
    Clay = auto()
    Loam = auto()  #
    Conglomerate = auto()
    Dacite = auto()  #
    Diorite = auto()
    Monzonite = auto()  #
    Dolostone = auto()
    Eclogite = auto()  #
    Evaporite = auto()
    Foidite = auto()  #
    Foidolite = auto()  #
    Gabbro = auto()  #
    Mylonite = auto()
    Gneiss = auto()
    Granite = auto()
    Granodiorite = auto()
    Granulite = auto()  #
    Schist = auto()
    Syenite = auto()  #
    Granofels = auto()  #
    Peridotite = auto()  #
    Pyroxenite = auto()  #
    Granophyre = auto()  #
    Hornfels = auto()  #
    Ignimbrite = auto()  #
    Kakirite = auto()
    Latite = auto()  #
    Limestone = auto()
    Marble = auto()
    Marl = auto()  #
    Marlstone = auto()
    Pebble = auto()  #
    Sand = auto()
    Silt = auto()  #
    Peat = auto()  #
    Phyllite = auto()
    Pegmatite = auto()  #
    Siltstone = auto()
    Serpentinite = auto()
    Phonolite = auto()  #
    Prasinite = auto()  #
    Sandstone = auto()
    Pseudotachyllite = auto()  #
    Quartzite = auto()
    Rauwacke = auto()  #
    Rodingite = auto()  #
    Tonalite = auto()  #
    Tephrite = auto()  #
    Trachyte = auto()  #


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
