"""Ground truth data classes for the stratigraphy benchmark."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter, field_validator, model_serializer

from swissgeol_doc_processing.utils.file_utils import parse_text

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
    end: float | None = None
    start: float | None = None


class GroundTruthLayer(BaseModel):
    """A single annotated borehole layer."""

    model_config = {"extra": "forbid"}
    consolidated: GroundTruthConsolidated | None = None
    depth_interval: GroundTruthLayerDepth
    material_description: str | None = None
    unconsolidated: GroundTruthUnconsolidated | None = None

    @field_validator("material_description", mode="before")
    @classmethod
    def preprocess(cls, value: str | None) -> str:
        return parse_text(value) if value else None


class GroundTruthGroundwater(BaseModel):
    """A recorded groundwater measurement."""

    model_config = {"extra": "forbid"}
    date: str | None = None
    depth: float
    elevation: float


class GroundTruthCoordinates(BaseModel):
    """Coordinate / location of the borehole, as easting (E) and northing (N)."""

    model_config = {"extra": "forbid"}

    E: float
    N: float


class GroundTruthMetadata(BaseModel):
    """Borehole metadata extracted from the document header."""

    model_config = {"extra": "forbid"}
    coordinates: GroundTruthCoordinates | None = None
    drilling_date: str | None = None
    drilling_methods: list[str] | None = None
    original_name: str | None = None
    project_name: str | None = None
    reference_elevation: float | None = None
    total_depth: float | None = None


class GroundTruthBorehole(BaseModel):
    """Ground truth data for a single borehole."""

    model_config = {"extra": "forbid"}
    borehole_index: int
    groundwater: list[GroundTruthGroundwater] | None
    layers: list[GroundTruthLayer]
    metadata: GroundTruthMetadata


class GroundTruth:
    """Ground truth data for the stratigraphy benchmark."""

    def __init__(self, path: Path) -> None:
        """Instanciate the GroundTruth object.

        Args:
            path (Path): the path to the Ground truth file
        """
        self.path = path
        self.ground_truth = defaultdict(lambda: defaultdict(dict))

        # Load the ground truth data
        with open(path, encoding="utf-8") as in_file:
            ground_truth = json.load(in_file)

        # Validate entries
        for filename, data in ground_truth.items():
            self.ground_truth[filename] = TypeAdapter(list[GroundTruthBorehole]).validate_python(data)

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
        return {}

    def to_json(self) -> dict:
        """Convert ground truth objects to dict."""
        return {
            filename: [borehole.model_dump() for borehole in boreholes]
            for filename, boreholes in self.ground_truth.items()
        }
