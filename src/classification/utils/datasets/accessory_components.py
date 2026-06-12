"""Accessory components classification dataset module."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem

logger = logging.getLogger(__name__)


class AccessoryComponentsSystem(ClassificationSystem):
    """Classification system for accessory components of consolidated geological layers."""

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize an accessory components class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "fish remains").

        Returns:
            str: The normalized accessory components class string (e.g., "fish_remains").
        """
        return class_str.lower().replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[AccessoryComponentsClasses]:
        """Return the AccessoryComponentsClasses Enum."""
        return cls.AccessoryComponentsClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "accessory_components"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["consolidated", "accessory_components"]

    @classmethod
    def get_default_class_value(cls) -> AccessoryComponentsClasses:
        """Return the default value for the enum class."""
        return cls.AccessoryComponentsClasses.not_specified

    @classmethod
    def is_multi_label(cls) -> bool:
        return True

    class AccessoryComponentsClasses(IntEnum):
        """Accessory component classes for consolidated soil classification (0-based indexing)."""

        not_specified = 0
        algae = auto()
        algal_mats = auto()
        ammonites = auto()
        aptychi = auto()
        ash = auto()
        belemnites = auto()
        bioclasts = auto()
        biodetritus = auto()
        bitumen = auto()
        bivalves = auto()
        brachiopods = auto()
        bryozoans = auto()
        calcareous_concretion = auto()
        calcareous_oncoids = auto()
        calcareous_ooids = auto()
        calcareous_pisoids = auto()
        calpionellids = auto()
        cephalopods = auto()
        chert = auto()
        coal_fragments = auto()
        coccoliths = auto()
        concretion = auto()
        corals = auto()
        crinoids = auto()
        crystals = auto()
        diatoms = auto()
        dinoflagellates = auto()
        echinoderms = auto()
        echinoids = auto()
        fish_remains = auto()
        foraminifera = auto()
        fossils = auto()
        gastropods = auto()
        gryphaea = auto()
        iron_ooids = auto()
        iron_pisoids = auto()
        lapilli = auto()
        lithoclasts = auto()
        molluscs = auto()
        nautilids = auto()
        nummulites = auto()
        oncoids = auto()
        ooids = auto()
        organic_matter = auto()
        ostracods = auto()
        other = auto()
        oysters = auto()
        pellets = auto()
        pisoids = auto()
        plant_remains = auto()
        plant_root_tubes = auto()
        pollen = auto()
        pyroclast = auto()
        radiolarians = auto()
        rootlets = auto()
        rudists = auto()
        sideritic_concretion = auto()
        silicified_wood = auto()
        spicules = auto()
        sponges = auto()
        spores = auto()
        stromatolites = auto()
        tintinnids = auto()
        vertebrates = auto()
        wood_fragments = auto()
