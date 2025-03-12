"""Classes for the predictions per borehole, and for associating them with ground truth data."""

import dataclasses

from stratigraphy.depths_materials_column_pairs.bounding_boxes import BoundingBoxes
from stratigraphy.groundwater.groundwater_extraction import GroundwatersInBorehole
from stratigraphy.layer.layer import LayersInBorehole
from stratigraphy.metadata.metadata import BoreholeMetadata


@dataclasses.dataclass
class BoreholePredictions:
    """Class that hold predicted information about a single borehole."""

    borehole_index: int
    layers_in_borehole: LayersInBorehole
    file_name: str
    metadata: BoreholeMetadata
    groundwater_in_borehole: GroundwatersInBorehole
    bounding_boxes: list[BoundingBoxes]

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "borehole_index": self.borehole_index,
            "metadata": self.metadata.to_json(),
            "layers": [layer.to_json() for layer in self.layers_in_borehole.layers],
            "bounding_boxes": [bboxes.to_json() for bboxes in self.bounding_boxes],
            "groundwater": self.groundwater_in_borehole.to_json() if self.groundwater_in_borehole is not None else [],
        }

    @classmethod
    def from_json(cls, json_object, file_name) -> "BoreholePredictions":
        """Extract a BoreholePrediction object from a json dictionary.

        Args:
            json_object (dict): the json object containing the informations of the borehole
            file_name: the file name

        Returns:
            (BoreholePredictions): the extracted object
        """
        return cls(
            json_object["borehole_index"],
            LayersInBorehole.from_json(json_object["layers"]),
            file_name,
            BoreholeMetadata.from_json(json_object["metadata"]),
            GroundwatersInBorehole.from_json(json_object["groundwater"]),
            [BoundingBoxes.from_json(bbox_json) for bbox_json in json_object["bounding_boxes"]],
        )


@dataclasses.dataclass
class BoreholePredictionsWithGroundTruth:
    """Predictions for a specific borehole with associated ground truth."""

    predictions: BoreholePredictions | None
    ground_truth: dict


@dataclasses.dataclass
class BoreholeLayersWithGroundTruth:
    """Stratigraphy predictions for a specific borehole with associated ground truth."""

    layers: LayersInBorehole | None
    ground_truth: list


@dataclasses.dataclass
class BoreholeGroundwaterWithGroundTruth:
    """Groundwater predictions for a specific borehole with associated ground truth."""

    groundwater: GroundwatersInBorehole | None
    ground_truth: list


@dataclasses.dataclass
class BoreholeMetadataWithGroundTruth:
    """Borehole metadata predictions for a specific borehole with associated ground truth."""

    metadata: BoreholeMetadata | None
    ground_truth: dict


@dataclasses.dataclass
class FilePredictionsWithGroundTruth:
    """All predictions for a specific file with associated ground truth."""

    filename: str
    language: str
    boreholes: list[BoreholePredictionsWithGroundTruth]


@dataclasses.dataclass
class FileMetadataWithGroundTruth:
    """All borehole metadata predictions for a specific file with associated ground truth."""

    filename: str
    boreholes: list[BoreholeMetadataWithGroundTruth]


@dataclasses.dataclass
class FileLayersWithGroundTruth:
    """All stratigraphy predictions for a specific file with associated ground truth."""

    filename: str
    language: str
    boreholes: list[BoreholeLayersWithGroundTruth]


@dataclasses.dataclass
class FileGroundwaterWithGroundTruth:
    """All groundwater predictions for a specific file with associated ground truth."""

    filename: str
    boreholes: list[BoreholeGroundwaterWithGroundTruth]
