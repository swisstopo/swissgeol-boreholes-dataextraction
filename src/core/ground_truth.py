"""Ground truth data classes for the stratigraphy benchmark."""

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter, model_serializer

from extraction.utils.json import JsonFloatDecimal

logger = logging.getLogger(__name__)


class ExcludeNoneBaseModel(BaseModel):
    """Avoid dumping all elements if none."""

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class GroundTruthConsolidated(ExcludeNoneBaseModel):
    """Ground truth material properties for consolidated geological layers."""

    model_config = {"extra": "forbid"}
    accessory_components: list[str] | None = None
    alteration_degree: str | None = None
    cementation: str | None = None
    lithology: str | None = None
    mineral_components: list[str] | None = None
    primary_color: str | None = None
    uscs: list[str] | None = None


class GroundTruthUnconsolidated(ExcludeNoneBaseModel):
    """Ground truth material properties for unconsolidated geological layers."""

    model_config = {"extra": "forbid"}
    alteration_degree: str | None = None
    debris: list[str] | None = None
    grain_angularity: list[str] | None = None
    grain_shape: list[str] | None = None
    main: str | None = None
    organic_components: list[str] | None = None
    other: list[str] | None = None
    primary_color: str | None = None
    uscs: list[str] | None = None


class GroundTruthLayerDepth(BaseModel):
    """Depth interval of a borehole layer."""

    model_config = {"extra": "forbid"}
    end: JsonFloatDecimal | None = None
    start: JsonFloatDecimal | None = None


class GroundTruthLayer(BaseModel):
    """A single annotated borehole layer."""

    model_config = {"extra": "forbid"}
    consolidated: GroundTruthConsolidated | None = None
    depth_interval: GroundTruthLayerDepth
    material_description: str | None = None
    unconsolidated: GroundTruthUnconsolidated | None = None


class GroundTruthGroundwater(BaseModel):
    """A recorded groundwater measurement."""

    model_config = {"extra": "forbid"}
    date: str | None = None
    depth: JsonFloatDecimal
    elevation: JsonFloatDecimal


class GroundTruthCoordinates(BaseModel):
    """Coordinate / location of the borehole, as easting (E) and northing (N)."""

    model_config = {"extra": "forbid"}

    E: JsonFloatDecimal
    N: JsonFloatDecimal


class GroundTruthMetadata(BaseModel):
    """Borehole metadata extracted from the document header."""

    model_config = {"extra": "forbid"}
    borehole_purpose: str | None = None
    borehole_type: str | None = None
    coordinates: GroundTruthCoordinates | None = None
    drilling_date: str | None = None
    drilling_methods: list[str] | None = None
    original_name: str | None = None
    project_name: str | None = None
    reference_elevation: JsonFloatDecimal | None = None
    total_depth: float | None = None


class GroundTruthBorehole(BaseModel):
    """Ground truth data for a single borehole."""

    model_config = {"extra": "forbid"}
    borehole_index: int
    groundwater: list[GroundTruthGroundwater] | None = None
    layers: list[GroundTruthLayer]
    metadata: GroundTruthMetadata


class GroundTruth(BaseModel):
    """Ground truth data for the stratigraphy benchmark."""

    ground_truth: dict[str, list[GroundTruthBorehole]]

    @classmethod
    def from_path(cls, path: Path) -> "GroundTruth":
        """Instantiate the GroundTruth object.

        Args:
            path (Path): the path to the Ground truth file
        """
        # Load the ground truth data
        with open(path, encoding="utf-8") as in_file:
            ground_truth = json.load(in_file, parse_float=Decimal)

        # Validate entries
        ground_truth_dict: dict[str, list[GroundTruthBorehole]] = {}
        for filename, data in ground_truth.items():
            ground_truth_dict[filename] = TypeAdapter(list[GroundTruthBorehole]).validate_python(data)
        return cls(ground_truth=ground_truth_dict)

    def for_file(self, file_name: str) -> list[GroundTruthBorehole]:
        """Get the ground truth data for a given file.

        Args:
            file_name (str): The file name.

        Returns:
            list[GroundTruthBorehole]: The ground truth data for the file.
        """
        if file_name in self.ground_truth:
            return self.ground_truth[file_name]

        logger.warning("No ground truth data found for %s.", file_name)
        return []

    @model_serializer(mode="wrap")
    def serialize_flat_json(self, handler) -> dict:
        """Serialize as a dict with the filename as key (assumed ot be unique)."""
        default_serialized = handler(self)
        return {filename: boreholes for filename, boreholes in default_serialized["ground_truth"].items()}
