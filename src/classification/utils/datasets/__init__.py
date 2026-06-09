"""Classification datasets package — exports all classification systems and ExistingClassificationSystems."""

from enum import Enum
from typing import Literal

from classification.utils.datasets.accessory_components import AccessoryComponentsSystem
from classification.utils.datasets.classification import ClassificationSystem
from classification.utils.datasets.color import ColorConsolidatedSystem, ColorSystem, ColorUnconsolidatedSystem
from classification.utils.datasets.debris import DebrisUnconsolidatedSystem
from classification.utils.datasets.en_main import ENMainSystem
from classification.utils.datasets.grain_angularity import GrainAngularitySystem
from classification.utils.datasets.lithology import LithologySystem
from classification.utils.datasets.mineral_components import MineralComponentsSystem
from classification.utils.datasets.organic_components import OrganicComponentsUnconsolidatedSystem
from classification.utils.datasets.uscs import USCSSystem


class ExistingClassificationSystems(Enum):
    """Enum listing all existing classification types.

    The value of each entry is the Classification system class, not an instance of the class.
    """

    accessory_components = AccessoryComponentsSystem
    color = ColorSystem
    color_consolidated = ColorConsolidatedSystem
    color_unconsolidated = ColorUnconsolidatedSystem
    debris = DebrisUnconsolidatedSystem
    en_main = ENMainSystem
    grain_angularity = GrainAngularitySystem
    lithology = LithologySystem
    mineral_components = MineralComponentsSystem
    organic_components = OrganicComponentsUnconsolidatedSystem
    uscs = USCSSystem

    @classmethod
    def get_classification_system_type(
        cls,
        class_system: Literal[
            "accessory_components",
            "color",
            "color_consolidated",
            "color_unconsolidated",
            "debris",
            "en_main",
            "grain_angularity",
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
