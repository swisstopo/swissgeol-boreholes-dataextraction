"""Layer class definition."""

from dataclasses import dataclass

import pymupdf

from extraction.features.utils.data_extractor import ExtractedFeature
from extraction.features.utils.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin
from extraction.features.utils.text.textblock import MaterialDescription
from utils.file_utils import parse_text

from ..interval.interval import Interval
from .page_bounding_boxes import PageBoundingBoxes


class LayerDepthsEntry(RectWithPageMixin):
    """Represents the upper or lower limit of a layer, used specifically for visualization and evaluation.

    Unlike `DepthColumnEntry` in `sidebarentry.py`, this class holds the extracted depth information,
    rather than being involved throughout the extraction process. It includes utility methods for
    data serialization, such as `to_json()`.
    """

    def __init__(self, value: float, rect: pymupdf.Rect, page_number: int):
        self.value = value
        self.rect_with_page = RectWithPage(rect, page_number)

    def to_json(self):
        """Convert the LayerDepthsEntry object to a JSON serializable format."""
        return {
            "value": self.value,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1] if self.rect else None,
            "page": self.page_number if self.page_number else None,
        }

    def __repr__(self):
        return f"{self.value}"

    @classmethod
    def from_json(cls, data: dict) -> "LayerDepthsEntry":
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the layer depths entry.

        Returns:
            DepthColumnEntry: the corresponding LayerDepthsEntry object.
        """
        return cls(value=data["value"], rect=pymupdf.Rect(data["rect"]), page_number=data["page"])


@dataclass
class LayerDepths:
    """Represents the start and end depth boundaries of a layer.

    Unlike the class `Interval` from `interval.py`, which is used in logical depth computations and extraction flow,
    the class `LayerDepths` is primarily used for data representation, visualiation and evaluation. It holds two
    extracted depth entries (`LayerDepthsEntry`), as opposed to Interval holding two `DepthColumnEntry`.
    """

    start: LayerDepthsEntry | None
    end: LayerDepthsEntry | None

    def get_line_anchor(self, page_number) -> pymupdf.Point | None:
        """Get the anchor point for the line connecting the start and end depths.

        Args:
            page_number (int): The page number for which to get the anchor point.

        Returns:
            pymupdf.Point | None: The anchor point for the line, or None if not applicable.
        """
        if self.start and self.end:
            if self.start.page_number == self.end.page_number:
                return pymupdf.Point(
                    max(self.start.rect.x1, self.end.rect.x1), (self.start.rect.y0 + self.end.rect.y1) / 2
                )
            else:
                if self.start.page_number == page_number:
                    return pymupdf.Point(self.start.rect.x1, self.start.rect.y1)
                elif self.end.page_number == page_number:
                    # anchor at the top of the page
                    return pymupdf.Point(self.end.rect.x1, 0)
                else:
                    return None
        elif self.start:
            return pymupdf.Point(self.start.rect.x1, self.start.rect.y1)
        elif self.end:
            return pymupdf.Point(self.end.rect.x1, self.end.rect.y0)

    def get_background_rect(self, page_number: int, page_height: float) -> pymupdf.Rect | None:
        """Get the background rectangle for the layer depths.

        Args:
            page_number (int): The page number for which to get the background rectangle.
            page_height (float): The height of the page.

        Returns:
            pymupdf.Rect | None: The background rectangle for the layer depths, or None if not applicable.
        """
        if not (self.start and self.end):
            return None
        if self.start.page_number != self.end.page_number:
            if page_number == self.start.page_number:
                return pymupdf.Rect(self.start.rect.x0, self.start.rect.y1, self.start.rect.x1, page_height)
            elif page_number == self.end.page_number:
                return pymupdf.Rect(self.end.rect.x0, 0, self.end.rect.x1, self.end.rect.y0)
            else:
                return None
        if self.start.rect.y1 < self.end.rect.y0:
            return pymupdf.Rect(
                self.start.rect.x0, self.start.rect.y1, max(self.start.rect.x1, self.end.rect.x1), self.end.rect.y0
            )
        return None

    def to_json(self):
        """Convert the LayerDepths object to a JSON serializable format."""
        return {"start": self.start.to_json() if self.start else None, "end": self.end.to_json() if self.end else None}

    @classmethod
    def from_json(cls, data: dict) -> "LayerDepths":
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionary representing the layer depths.

        Returns:
            DepthColumnEntry: the corresponding LayerDepths object.
        """
        return cls(
            start=LayerDepthsEntry.from_json(data["start"]) if data["start"] else None,
            end=LayerDepthsEntry.from_json(data["end"]) if data["end"] else None,
        )

    @classmethod
    def from_interval(cls, interval: Interval) -> "LayerDepths":
        """Converts an Interval to a LayerDepths object.

        Args:
            interval (Interval): an AAboveBInterval or AToBInterval.

        Returns:
            LayerDepths: the corresponding LayerDepths object.
        """
        start = interval.start
        end = interval.end
        return cls(
            start=LayerDepthsEntry(start.value, start.rect, start.page_number) if start else None,
            end=LayerDepthsEntry(end.value, end.rect, end.page_number) if end else None,
        )

    def is_valid_depth_interval(self, start: float, end: float) -> bool:
        """Validate if self and the depth interval start-end match.

        Args:
            start (float): The start value of the interval.
            end (float): The end value of the interval.

        Returns:
            bool: True if the depth intervals match, False otherwise.
        """
        if self.start is None:
            return (start == 0) and (end == self.end.value)

        if (self.start is not None) and (self.end is not None):
            return start == self.start.value and end == self.end.value

        return False


@dataclass
class Layer(ExtractedFeature):
    """Represents a finalized layer prediction in a borehole profile.

    A `Layer` combines a material description with its associated depth information,
    typically derived from a cleaned and validated `Sidebar` after extraction. It is the
    main data structure used for representing stratigraphy in the `ExtractedBorehole` class.
    """

    material_description: MaterialDescription
    depths: LayerDepths | None

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return f"Layer(material_description={self.material_description}, depths={self.depths})"

    def description_nonempty(self) -> bool:
        return parse_text(self.material_description.text) != ""

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "material_description": self.material_description.to_json() if self.material_description else None,
            "depths": self.depths.to_json() if self.depths else None,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Layer":
        """Converts a dictionary to an object.

        Args:
            data (dict): A dictionarie representing the layer.

        Returns:
            list[LayerPrediction]: A list of LayerPrediction objects.
        """
        material_prediction = MaterialDescription.from_json(data["material_description"])
        depths = LayerDepths.from_json(data["depths"]) if ("depths" in data and data["depths"] is not None) else None

        return Layer(material_description=material_prediction, depths=depths)


@dataclass
class LayersInBorehole:
    """Represent the data for all layers in a borehole profile."""

    layers: list[Layer]

    def to_json(self):
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return [layer.to_json() for layer in self.layers]

    @classmethod
    def from_json(cls, json_object) -> "LayersInBorehole":
        """Extract a LayersInBorehole object from a json dictionary.

        Args:
            json_object (dict): The object as a dictionary.

        Returns:
            LayersInBorehole: The LayersInBorehole object.
        """
        return cls([Layer.from_json(layer_data) for layer_data in json_object])


@dataclass
class ExtractedBorehole:
    """A class to store the extracted information of one single borehole."""

    predictions: list[Layer]
    bounding_boxes: list[PageBoundingBoxes]  # one for each page that the borehole spans


class LayersInDocument:
    """A class to represent predictions for a single document.

    It contains a list of LayersInBorehole, not just a list of Layer.
    """

    def __init__(self, boreholes_layers: list[ExtractedBorehole], filename: str):
        self.boreholes_layers_with_bb = boreholes_layers
        self.filename = filename

    def assign_layers_to_boreholes(self, layer_predictions: list[ExtractedBorehole]):
        """SIMPLIFICATION: currently assumes that if there is more than one page, there is a single borehole.

        This means that only pdfs with one page could contain more than one borehole.
        This is the case for all example from zurich and geoquat,validation.
        If we want to be more robust, a matching should be done to determine which layers goes with which borehole.

        Args:
            layer_predictions (list[ExtractedBorehole]): List containing all boreholes with all layers
        """
        if not layer_predictions:
            return
        if not self.boreholes_layers_with_bb:
            # first page
            self.boreholes_layers_with_bb = layer_predictions
        else:
            # second page, use assumption, also for the bounding boxes
            main_borehole = self.boreholes_layers_with_bb[0]
            borehole_continuation = layer_predictions[0]
            if (
                (main_borehole.predictions and main_borehole.predictions[-1].depths)  # ensure depths exist
                and main_borehole.predictions[-1].depths.start is not None  # start value is set
                and main_borehole.predictions[-1].depths.end is None  # end value is None
                and (borehole_continuation.predictions and borehole_continuation.predictions[0].depths)  # depths exist
                and borehole_continuation.predictions[0].depths.start is None  # start value is None
                and borehole_continuation.predictions[0].depths.end is not None  # end value is not None
            ):
                # if the last interval of the previous page is open-ended, and the first of this page has no start
                # value, it probably means that they refer to the same layer.
                main_borehole.predictions[-1] = self._build_spanning_layer(borehole_continuation.predictions[0])
                borehole_continuation.predictions = borehole_continuation.predictions[1:]
            main_borehole.bounding_boxes.extend(borehole_continuation.bounding_boxes)
            main_borehole.predictions.extend(borehole_continuation.predictions)

    def _build_spanning_layer(self, first_next_layer: Layer) -> Layer:
        last_prev_layer = self.boreholes_layers_with_bb[0].predictions[-1]
        return Layer(
            material_description=MaterialDescription(
                text=last_prev_layer.material_description.text + " " + first_next_layer.material_description.text,
                lines=last_prev_layer.material_description.lines + first_next_layer.material_description.lines,
            ),
            depths=LayerDepths(start=last_prev_layer.depths.start, end=first_next_layer.depths.end),
        )
