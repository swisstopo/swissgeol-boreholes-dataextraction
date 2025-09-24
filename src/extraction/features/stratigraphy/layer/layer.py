"""Layer class definition."""

from dataclasses import dataclass

import numpy as np
import pymupdf

from extraction.features.utils.data_extractor import ExtractedFeature
from extraction.features.utils.geometry.geometry_dataclasses import RectWithPage, RectWithPageMixin
from extraction.features.utils.geometry.util import x_overlap_significant_smallest
from extraction.features.utils.text.textblock import MaterialDescription
from utils.file_utils import parse_text

from ..interval.interval import Interval
from .page_bounding_boxes import PageBoundingBoxes

DEPTHS_QUANTILE_SLACK = 0.1
SIDEBAR_BBOX_OVERLAP = 0.5
MAT_DESCR_BBOX_OVERLAP = 0.9


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
        """Assign boreholes from the predictions to the existing boreholes.

        This method will attempt to match the predicted boreholes to the existing boreholes based on their positions
        and continuity across pages. If a match is found, the layers are merged; otherwise, new boreholes are added.

        Args:
            layer_predictions (list[ExtractedBorehole]): List containing all boreholes with all layers

        Note:
            This method modifies the state of the object by updating the boreholes_layers_with_bb attribute.
        """
        # 0. filtering of empty objects
        if not layer_predictions or not any(lay.predictions for lay in layer_predictions):
            return
        if not self.boreholes_layers_with_bb:
            self.boreholes_layers_with_bb = layer_predictions
            return

        current_page = layer_predictions[0].bounding_boxes[0].page  # all the layer_predictions come from one page

        # 1. Out of the previously detected boreholes, determine which one should be continued.
        borehole_to_extend = self._identify_borehole_to_extend(current_page)
        if borehole_to_extend is None:
            self.boreholes_layers_with_bb.extend(layer_predictions)
            return

        # 2. If there is more than one borehole on the current page, determine which one should extend the previous
        borehole_continuation = self._identify_borehole_continuation(layer_predictions)
        remaining_boreholes = [borehole for borehole in layer_predictions if borehole is not borehole_continuation]

        # 3. Is the the current borehole likelly to be the continuation of the previous.
        if not self._is_continuation(borehole_to_extend, borehole_continuation, current_page):
            self.boreholes_layers_with_bb.extend(layer_predictions)
            return

        # 4. Combine the boreholes and deal with the remaining predictions
        self._merge_boreholes(borehole_to_extend, borehole_continuation)
        if remaining_boreholes:
            self.boreholes_layers_with_bb.extend(remaining_boreholes)

    def _identify_borehole_to_extend(self, current_page: int) -> ExtractedBorehole | None:
        """Identify the borehole to potentially extend with the layers on this page.

        If there is more than one previous borehole, we rank them to determine the most likely to be continued. The
        score uses y1 (lower means more likely to continue on the next page) and also the width, so that
        narrow parasite boreholes don't get too much weight.

        Args:
            current_page (int): The current page number.

        Returns:
            ExtractedBorehole | None: The borehole to extend, or None if no borehole is found on the previous page.
        """
        assert current_page > 1, "Can't be here on the first page"

        boreholes_with_score = []
        for borehole in self.boreholes_layers_with_bb:
            prev_page_bbox = next((bbox for bbox in borehole.bounding_boxes if bbox.page == current_page - 1), None)
            if not prev_page_bbox:
                continue

            outer_rect = prev_page_bbox.get_outer_rect()  # get the boreholes outer Rect
            boreholes_with_score.append((borehole, outer_rect.y1 + outer_rect.width))

        if not boreholes_with_score:
            # no boreholes on the previous page, the ones detected here can't be the continuation of any
            return None

        boreholes_with_score.sort(key=lambda x: x[1], reverse=True)  # highest score first
        return boreholes_with_score[0][0]

    def _identify_borehole_continuation(self, layer_predictions: list[ExtractedBorehole]) -> ExtractedBorehole:
        """Identify the borehole on the current page that is most likely to be the continuation of the previous.

        Args:
            layer_predictions (list[ExtractedBorehole]): List containing all detected boreholes on this page.

        Returns:
            ExtractedBorehole: The borehole that is most likely to be the continuation of the previous.
        """
        return min(
            [(borehole, borehole.bounding_boxes[0].get_outer_rect().y0) for borehole in layer_predictions],
            key=lambda x: x[1],  # sort by y0 value
        )[0]  # grab the borehole with the smallest y0

    def _is_continuation(
        self, borehole_to_extend: ExtractedBorehole, borehole_continuation: ExtractedBorehole, current_page: int
    ) -> bool:
        """Determine if the borehole on the current page is the continuation of the previous borehole.

        This method check if the depths are continuous, if the sidebar bounding boxes overlap or if the material
        description bounding boxes significantly.

        Args:
            borehole_to_extend (ExtractedBorehole): The borehole from the previous page
            borehole_continuation (ExtractedBorehole): The borehole from the current page
            current_page (int): The current page number

        Returns:
            bool: True if the current borehole is the continuation of the previous borehole, False otherwise.
        """
        ok_prev_layers = [lay for lay in borehole_to_extend.predictions if lay.depths is not None]
        prev_depths = [d.value for lay in ok_prev_layers for d in (lay.depths.start, lay.depths.end) if d is not None]

        ok_layers = [lay for lay in borehole_continuation.predictions if lay.depths is not None]
        depths = [d.value for lay in ok_layers for d in (lay.depths.start, lay.depths.end) if d is not None]
        if (
            depths
            and prev_depths
            and np.quantile(depths, DEPTHS_QUANTILE_SLACK) >= np.quantile(prev_depths, 1 - DEPTHS_QUANTILE_SLACK)
        ):
            # use quantile to allow some slack (e.g. few undetected duplicated layers)
            return True  # if depths are continue, it is a continuation

        prev_sidebar_bbox = next(
            bbox.sidebar_bbox for bbox in borehole_to_extend.bounding_boxes if bbox.page == current_page - 1
        )
        current_sidebar_bbox = borehole_continuation.bounding_boxes[0].sidebar_bbox
        if (
            prev_sidebar_bbox is not None
            and current_sidebar_bbox is not None
            and x_overlap_significant_smallest(prev_sidebar_bbox.rect, current_sidebar_bbox.rect, SIDEBAR_BBOX_OVERLAP)
        ):
            return True  # if sidebars overlaps slighly, it is a continuation

        prev_mat_bbox = next(
            bbox.material_description_bbox
            for bbox in borehole_to_extend.bounding_boxes
            if bbox.page == current_page - 1
        )
        current_mat_bbox = borehole_continuation.bounding_boxes[0].material_description_bbox
        return (
            prev_mat_bbox is not None
            and current_mat_bbox is not None
            and x_overlap_significant_smallest(prev_mat_bbox.rect, current_mat_bbox.rect, MAT_DESCR_BBOX_OVERLAP)
        )  # if material bounding box overlaps significantly, it is a continuation

    def _merge_boreholes(self, borehole_to_extend: ExtractedBorehole, borehole_continuation: ExtractedBorehole):
        """Merge the layers of the current borehole into the borehole to extend.

        If the last layer of the previous borehole and the first layer of the current borehole appear to be the same
        layer (based on depth continuity), they are merged into a single layer that spans both pages.

        Args:
            borehole_to_extend (ExtractedBorehole): The borehole from the previous page
            borehole_continuation (ExtractedBorehole): The borehole from the current page

        Note:
            This method modifies the state of the borehole_to_extend object by updating its predictions and bounding
        """
        # Do the last layer of the previous borehole and the first of the current belong to the same layer.
        last_prev_depths = borehole_to_extend.predictions[-1].depths if borehole_to_extend.predictions else None
        first_depths = borehole_continuation.predictions[0].depths if borehole_continuation.predictions else None

        if (
            (last_prev_depths and last_prev_depths.start and last_prev_depths.end is None)
            and (first_depths and first_depths.start is None and first_depths.end)
            and (last_prev_depths.start.value < first_depths.end.value)
        ):
            # if the last interval of the previous page is open-ended, and the first of this page has no start
            # value, it probably means that they refer to the same layer.
            borehole_to_extend.predictions[-1] = self._build_spanning_layer(borehole_continuation.predictions[0])
            borehole_continuation.predictions = borehole_continuation.predictions[1:]
        borehole_to_extend.bounding_boxes.extend(borehole_continuation.bounding_boxes)
        borehole_to_extend.predictions.extend(borehole_continuation.predictions)

    def _build_spanning_layer(self, first_next_layer: Layer) -> Layer:
        """Build a new layer that spans the last layer of the previous borehole and the first of the current.

        Args:
            first_next_layer (Layer): The first layer of the current borehole.

        Returns:
            Layer: A new layer that contains the material description of both layers and spans their depths.
        """
        last_prev_layer = self.boreholes_layers_with_bb[0].predictions[-1]
        return Layer(
            material_description=MaterialDescription(
                text=last_prev_layer.material_description.text + " " + first_next_layer.material_description.text,
                lines=last_prev_layer.material_description.lines + first_next_layer.material_description.lines,
            ),
            depths=LayerDepths(start=last_prev_layer.depths.start, end=first_next_layer.depths.end),
        )
