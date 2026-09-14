"""Classification datasets package — exports all classification systems and ExistingClassificationSystems."""

from enum import Enum
from typing import Literal

from classification.utils.datasets.accessory_components import AccessoryComponentsSystem
from classification.utils.datasets.alteration_degree import (
    AlterationDegreeConsolidatedSystem,
    AlterationDegreeSystem,
    AlterationDegreeUnconsolidatedSystem,
)
from classification.utils.datasets.borehole_type import BoreholeTypeSystem
from classification.utils.datasets.cementation import CementationSystem
from classification.utils.datasets.classification import ClassificationSystem
from classification.utils.datasets.color import ColorConsolidatedSystem, ColorSystem, ColorUnconsolidatedSystem
from classification.utils.datasets.debris import DebrisSystem
from classification.utils.datasets.en_main import ENMainSystem, ENSecondarySystem
from classification.utils.datasets.grain_angularity import GrainAngularitySystem
from classification.utils.datasets.grain_shape import GrainShapeSystem
from classification.utils.datasets.lithology import LithologySystem
from classification.utils.datasets.mineral_components import MineralComponentsSystem
from classification.utils.datasets.organic_components import OrganicComponentsSystem
from classification.utils.datasets.uscs import USCSSystem


class ExistingClassificationSystems(Enum):
    """Enum listing all existing classification types.

    The value of each entry is the Classification system class, not an instance of the class.
    """

    accessory_components = AccessoryComponentsSystem
    alteration_degree = AlterationDegreeSystem
    alteration_degree_consolidated = AlterationDegreeConsolidatedSystem
    alteration_degree_unconsolidated = AlterationDegreeUnconsolidatedSystem
    borehole_type = BoreholeTypeSystem
    cementation = CementationSystem
    color = ColorSystem
    color_consolidated = ColorConsolidatedSystem
    color_unconsolidated = ColorUnconsolidatedSystem
    debris = DebrisSystem
    en_main = ENMainSystem
    en_secondary = ENSecondarySystem
    grain_angularity = GrainAngularitySystem
    grain_shape = GrainShapeSystem
    lithology = LithologySystem
    mineral_components = MineralComponentsSystem
    organic_components = OrganicComponentsSystem
    uscs = USCSSystem

    @classmethod
    def get_classification_system_type(
        cls,
        class_system: Literal[
            "accessory_components",
            "alteration_degree",
            "alteration_degree_consolidated",
            "alteration_degree_unconsolidated",
            "borehole_type",
            "cementation",
            "color",
            "color_consolidated",
            "color_unconsolidated",
            "debris",
            "en_main",
            "en_secondary",
            "grain_angularity",
            "grain_shape",
            "lithology",
            "mineral_components",
            "organic_components",
            "uscs",
        ],
    ) -> type[ClassificationSystem]:
        """Returns the class of a classification system based on input string.

        Args:
            class_system (Literal): The name of the classification system.

        Returns:
            Type[ClassificationSystem]: The associated ClassificationSystem class.

        """
        return cls[class_system].value
