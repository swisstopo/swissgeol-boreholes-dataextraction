"""Layer class definition."""

import uuid
from dataclasses import dataclass, field

import fitz
from stratigraphy.depthcolumn.depthcolumnentry import DepthColumnEntry
from stratigraphy.lines.line import TextLine, TextWord
from stratigraphy.text.textblock import MaterialDescription, TextBlock
from stratigraphy.util.interval import AnnotatedInterval, BoundaryInterval, Interval
from stratigraphy.util.util import parse_text


@dataclass
class Layer:
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

    @staticmethod
    def from_json(json_dict: dict) -> list["Layer"]:
        """Converts a dictionary to an object.

        Args:
            json_dict (dict): The dictionary to convert.

        Returns:
            LayerPrediction: The object.
        """
        page_layer_predictions_list: list[Layer] = []

        # Extract the layer predictions.
        for layer in json_dict:
            material_prediction = _create_textblock_object(layer["material_description"]["lines"])
            if "depth_interval" in layer:
                start = (
                    DepthColumnEntry(
                        value=layer["depth_interval"]["start"]["value"],
                        rect=fitz.Rect(layer["depth_interval"]["start"]["rect"]),
                        page_number=layer["depth_interval"]["start"]["page"],
                    )
                    if layer["depth_interval"]["start"] is not None
                    else None
                )
                end = (
                    DepthColumnEntry(
                        value=layer["depth_interval"]["end"]["value"],
                        rect=fitz.Rect(layer["depth_interval"]["end"]["rect"]),
                        page_number=layer["depth_interval"]["end"]["page"],
                    )
                    if layer["depth_interval"]["end"] is not None
                    else None
                )

                depth_interval_prediction = BoundaryInterval(start=start, end=end)
                layer_predictions = Layer(
                    material_description=material_prediction, depth_interval=depth_interval_prediction
                )
            else:
                layer_predictions = Layer(material_description=material_prediction, depth_interval=None)

            page_layer_predictions_list.append(layer_predictions)

        return page_layer_predictions_list


def _create_textblock_object(lines: dict) -> TextBlock:
    """Creates a TextBlock object from a dictionary.

    Args:
        lines (dict): The dictionary to convert.

    Returns:
        TextBlock: The object.
    """
    lines = [TextLine([TextWord(**line)]) for line in lines]
    return TextBlock(lines)


# @dataclass(kw_only=True)
# TODO: check if this makes more sense: class LayersOnPage(ExtractedFeature):
@dataclass
class LayersOnPage:
    """A class to represent predictions for a single page."""

    layers_on_page: list[Layer]

    def remove_empty_predictions(self):
        """Remove empty predictions from the predictions dictionary.

        Args:
            predictions (dict): Predictions dictionary.

        Returns:
            dict: Predictions dictionary without empty predictions.
        """
        for layer in self.layers_on_page:
            if parse_text(layer.material_description.text) == "":
                self.layers_on_page.remove(layer)


@dataclass
class LayersInDocument:
    """A class to represent predictions for a single document."""

    layers_in_document: list[LayersOnPage]
    filename: str

    def add_layers_on_page(self, layers_on_page: LayersOnPage):
        """Add layers on a page to the layers in the document.

        Args:
            layers_on_page (LayersOnPage): The layers on a page to add.
        """
        self.layers_in_document.append(layers_on_page)

    def get_all_layers(self) -> list[Layer]:
        """Get all layers in the document.

        Returns:
            list[Layer]: All layers in the document.
        """
        all_layers = []
        for layers_on_page in self.layers_in_document:
            all_layers.extend(layers_on_page.layers_on_page)
        return all_layers


@dataclass
class IntervalBlockGroup:
    """A class to represent a group of depth interval blocks."""

    depth_interval: Interval | list[Interval] | None
    block: TextBlock | list[TextBlock]
