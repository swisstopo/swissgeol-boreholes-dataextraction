"""Color classification system for consolidated and unconsolidated soils (primary_color field)."""

from __future__ import annotations

import logging
from enum import IntEnum, auto

from classification.utils.datasets.classification import ClassificationSystem

logger = logging.getLogger(__name__)


class ColorSystem(ClassificationSystem):
    """Base classification system for the primary color of geological layers.

    Use ColorConsolidatedSystem or ColorUnconsolidatedSystem — do not instantiate directly.
    """

    @classmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize a color class string.

        Args:
            class_str (str): The class string to be normalized (e.g. "dark grey").

        Returns:
            str: The normalized color class string (e.g., "dark_grey").
        """
        return class_str.lower().replace(" ", "_")

    @classmethod
    def get_enum(cls) -> type[ColorClasses]:
        """Return the ColorClasses Enum."""
        return cls.ColorClasses

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system."""
        return "color"

    @classmethod
    def get_default_class_value(cls) -> ColorClasses:
        """Default value for the enum (not specified)."""
        return cls.ColorClasses.not_specified

    class ColorClasses(IntEnum):
        """Complete color class list (0-based indexing)."""

        beige = 0
        beige_grey = auto()
        black = auto()
        bluish_grey = auto()
        bluish_red = auto()
        brown = auto()
        brownish_grey = auto()
        brownish_red = auto()
        brownish_yellow = auto()
        dark_beige = auto()
        dark_brown = auto()
        dark_green = auto()
        dark_grey = auto()
        dark_olive = auto()
        dark_red = auto()
        dark_yellow = auto()
        green = auto()
        greenish_beige = auto()
        greenish_grey = auto()
        grey = auto()
        greyish_beige = auto()
        greyish_blue = auto()
        greyish_brown = auto()
        greyish_green = auto()
        light_beige = auto()
        light_brown = auto()
        light_green = auto()
        light_grey = auto()
        light_olive = auto()
        light_yellow = auto()
        magenta = auto()
        not_specified = auto()
        ochre = auto()
        olive = auto()
        olive_yellow = auto()
        red = auto()
        reddish_brown = auto()
        reddish_ochre = auto()
        white = auto()
        yellow = auto()
        yellowish_brown = auto()
        yellowish_white = auto()


class ColorConsolidatedSystem(ColorSystem):
    """Classification system for the primary color of consolidated geological layers."""

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "color_consolidated"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["consolidated", "primary_color"]


class ColorUnconsolidatedSystem(ColorSystem):
    """Classification system for the primary color of unconsolidated geological layers."""

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        return "color_unconsolidated"

    @classmethod
    def get_layer_ground_truth_keys(cls) -> list[str]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        return ["unconsolidated", "primary_color"]
