"""Module for the main lithology classes."""

import logging
from enum import IntEnum, auto

logger = logging.getLogger(__name__)


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
    Andesite = auto()
    Aplite = auto()
    Basalt = auto()
    Basanite = auto()
    Bentonite = auto()
    Claystone = auto()
    Breccia = auto()
    Rhyolite = auto()
    Tuffite = auto()
    Cataclastite = auto()
    Clay = auto()
    Loam = auto()
    Conglomerate = auto()
    Dacite = auto()
    Diorite = auto()
    Monzonite = auto()
    Dolostone = auto()
    Eclogite = auto()
    Evaporite = auto()
    Foidite = auto()
    Foidolite = auto()
    Gabbro = auto()
    Mylonite = auto()
    Gneiss = auto()
    Granite = auto()
    Granodiorite = auto()
    Granulite = auto()
    Schist = auto()
    Syenite = auto()
    Granofels = auto()
    Peridotite = auto()
    Pyroxenite = auto()
    Granophyre = auto()
    Hornfels = auto()
    Ignimbrite = auto()
    Kakirite = auto()
    Latite = auto()
    Limestone = auto()
    Marble = auto()
    Marl = auto()
    Marlstone = auto()
    Pebble = auto()
    Sand = auto()
    Silt = auto()
    Peat = auto()
    Phyllite = auto()
    Pegmatite = auto()
    Siltstone = auto()
    Serpentinite = auto()
    Phonolite = auto()
    Prasinite = auto()
    Sandstone = auto()
    Pseudotachyllite = auto()
    Quartzite = auto()
    Rauwacke = auto()
    Rodingite = auto()
    Tonalite = auto()
    Tephrite = auto()
    Trachyte = auto()


def map_most_similar_lithology(lithology_str: str) -> LithologyClasses:
    """Maps the ground truth string to one of the LithologyClasses.

    Args:
        lithology_str (str): the ground truth string

    Returns:
        LithologyClasses: the matching class
    """
    potential_classes = []
    for class_ in LithologyClasses:
        index = lithology_str.lower().find(class_.name.lower())
        if index != -1:
            potential_classes.append((index, -len(class_.name.lower()), class_))

    if not potential_classes:
        logger.warning(
            f"{lithology_str} does have a matching class, mapping it to {LithologyClasses.kA.name} instead."
        )
        return LithologyClasses.kA  # keine Angabe = no indication

    potential_classes.sort()  # sort by position of match and length of pattern.
    if len(potential_classes) > 1:
        logger.warning(
            f"{lithology_str} has multiple matches ({[v[2].name for v in potential_classes]}),"
            f" mapping it to {potential_classes[0][2].name}"
        )
    return potential_classes[0][2]
