"""Layer class definition."""

import uuid
from dataclasses import dataclass, field

from stratigraphy.text.textblock import MaterialDescription, TextBlock
from stratigraphy.util.interval import AnnotatedInterval, BoundaryInterval


@dataclass
class LayerPrediction:
    """A class to represent predictions for a single layer."""

    material_description: TextBlock | MaterialDescription
    depth_interval: BoundaryInterval | AnnotatedInterval | None
    material_is_correct: bool = None
    depth_interval_is_correct: bool = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return (
            f"LayerPrediction(material_description={self.material_description}, depth_interval={self.depth_interval})"
        )

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "material_description": self.material_description.to_json() if self.material_description else None,
            "depth_interval": self.depth_interval.to_json() if self.depth_interval else None,
            "material_is_correct": self.material_is_correct,
            "depth_interval_is_correct": self.depth_interval_is_correct,
            "id": str(self.id),
        }
