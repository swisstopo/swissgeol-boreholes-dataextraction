"""Classification datasets package — exports all classification systems and ExistingClassificationSystems."""

from enum import Enum
from typing import Literal

from classification.utils.datasets.classification import ClassificationSystem
from classification.utils.datasets.color import ColorConsolidatedSystem, ColorSystem, ColorUnconsolidatedSystem
from classification.utils.datasets.en_main import ENMainSystem
from classification.utils.datasets.lithology import LithologySystem
from classification.utils.datasets.organic_components import OrganicComponentsUnconsolidatedSystem
from classification.utils.datasets.uscs import USCSSystem


class ExistingClassificationSystems(Enum):
    """Enum listing all existing classification types.

    The value of each entry is the Classification system class, not an instance of the class.
    """

    uscs = USCSSystem
    lithology = LithologySystem
    en_main = ENMainSystem
    color = ColorSystem
    color_consolidated = ColorConsolidatedSystem
    color_unconsolidated = ColorUnconsolidatedSystem
    organic_components = OrganicComponentsUnconsolidatedSystem

    @classmethod
    def get_classification_system_type(
        cls,
        class_system: Literal[
            "uscs", "lithology", "en_main", "color", "organic_components", "color_consolidated", "color_unconsolidated"
        ],
    ) -> type[ClassificationSystem]:
        """Returns the class of a classification system based on input string.

        Args:
            class_system (Literal): The name of the classification system.

        Returns:
            Type[ClassificationSystem]: The associated ClassificationSystem class.

        """
        return cls[class_system].value
