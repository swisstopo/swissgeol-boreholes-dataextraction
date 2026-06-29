"""Mineral classification dataset module."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem, ClassificationTask

logger = logging.getLogger(__name__)


class MineralComponentsSystem(ClassificationSystem):
    """Classification system for mineral components of consolidated geological layers."""

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a mineral class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "not specified", "K-feldspar").

        Returns:
            str: The normalized mineral class string (e.g. "not_specified", "k_feldspar").
        """
        return class_str.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[MineralComponents]:
        """Return the MineralComponents Enum."""
        return cls.MineralComponents

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system."""
        return "mineral_components"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[list[str]]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return [["consolidated", "mineral_components"]]

    @classmethod
    def get_default_class_value(cls) -> MineralComponents:
        """Return the default value for the enum class."""
        return cls.MineralComponents.not_specified

    @classmethod
    def classification_task(cls) -> ClassificationTask:
        return ClassificationTask.multi_label

    class MineralComponents(IntEnum):
        """Mineral classes list (0-based indexing)."""

        not_specified = 0
        actinolite = auto()
        adularia = auto()
        albite = auto()
        almandine = auto()
        amphibole = auto()
        andalusite = auto()
        andesine = auto()
        anhydrite = auto()
        ankerite = auto()
        anorthite = auto()
        anorthoclase = auto()
        anthracite = auto()
        antigorite = auto()
        apatite = auto()
        aragonite = auto()
        arsenopyrite = auto()
        asbestos = auto()
        augite = auto()
        baryte = auto()
        bauxite = auto()
        biotite = auto()
        calcite = auto()
        cassiterite = auto()
        celestine = auto()
        chalcopyrite = auto()
        chlorite = auto()
        chloritoid = auto()
        chromite = auto()
        chrysotile = auto()
        coesite = auto()
        copper = auto()
        cordierite = auto()
        corundum = auto()
        diaspore = auto()
        diopside = auto()
        dolomite = auto()
        enstatite = auto()
        epidote = auto()
        feldspar = auto()
        fluorite = auto()
        forsterite = auto()
        fuchsite = auto()
        galena = auto()
        garnet = auto()
        gibbsite = auto()
        glauconite = auto()
        glaucophane = auto()
        goethite = auto()
        gold = auto()
        graphite = auto()
        grossular = auto()
        gypsum = auto()
        halite = auto()
        hematite = auto()
        hornblende = auto()
        illite = auto()
        ilmenite = auto()
        jadeite = auto()
        kaolinite = auto()
        k_feldspar = auto()  # K-feldspar
        kyanite = auto()
        leucite = auto()
        lignite = auto()
        limonite = auto()
        magnesite = auto()
        magnetite = auto()
        mica = auto()
        microcline = auto()
        monazite = auto()
        montmorillonite = auto()
        muscovite = auto()
        nepheline = auto()
        oligoclase = auto()
        olivine = auto()
        omphacite = auto()
        orthoclase = auto()
        phengite = auto()
        phlogopite = auto()
        plagioclase = auto()
        prehnite = auto()
        pyrite = auto()
        pyrope = auto()
        pyrophyllite = auto()
        pyroxene = auto()
        quartz = auto()
        rutile = auto()
        sapphirine = auto()
        scapolite = auto()
        sericite = auto()
        serpentine = auto()
        siderite = auto()
        sillimanite = auto()
        silver = auto()
        smectite = auto()
        spessartite = auto()
        sphalerite = auto()
        spinel = auto()
        staurolite = auto()
        stilpnomelane = auto()
        sulfur = auto()
        talc = auto()
        titanite = auto()
        tourmaline = auto()
        tremolite = auto()
        vermiculite = auto()
        wollastonite = auto()
        zeolite = auto()
        zircon = auto()
        zoisite = auto()
        other = auto()
