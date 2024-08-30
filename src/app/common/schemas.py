"""Schemas based on the database domain.

Each schema has a default version representing a full object,
as well as a patch version with all fields optional for patch operations.
"""

########################################################################################################################
### Creare pngs schema
########################################################################################################################

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field


class PNGResponse(BaseModel):
    """Response schema for the create_pngs endpoint."""

    png_urls: list[str]


########################################################################################################################
### Extract data schema
########################################################################################################################


class FormatTypes(Enum):
    """Enum for the format types."""

    TEXT = "text"
    NUMBER = "number"
    COORDINATES = "coordinates"
    ELEVATION = "elevation"


class BoundingBox(BaseModel):
    """Bounding box schema."""

    x0: float = Field(..., example=0.0)
    y0: float = Field(..., example=0.0)
    x1: float = Field(..., example=100.0)
    y1: float = Field(..., example=100.0)


class Coordinates(BaseModel):
    """Coordinates schema."""

    east: float = Field(..., example=1.0)
    north: float = Field(..., example=2.0)
    page: int = Field(..., example=1)
    spacial_reference_system: str = Field(..., example="LV95")


class ExtractDataRequest(ABC, BaseModel):
    """Request schema for the extract_data endpoint.

    Each field in the Pydantic model can have an example parameter, which provides an inline
    example for that specific field.
    """

    filename: str = Field(..., example="document.png")
    page_number: int = Field(..., example=1)  # 1-based index
    bbox: BoundingBox = Field(..., example={"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0})
    format: FormatTypes = Field(..., example=FormatTypes.COORDINATES.value)

    class Config:
        """Make it possible to define an example for the entire request model in the Swagger UI.

        The schema_extra attribute inside the Config class allows you to define a complete
        example for the entire request model.
        """

        json_schema_extra = {
            "example": {
                "filename": "pdfs/geoquat/train/10012.pdf",
                "page_number": 1,
                "bbox": {"x0": 0.0, "y0": 0.0, "x1": 200.0, "y1": 200.0},
                "format": "coordinates",  # Adjust this to match your actual FormatTypes
            }
        }


class ExtractDataResponse(ABC, BaseModel):
    """Response schema for the extract_data endpoint."""

    bbox: BoundingBox = Field(..., example={"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0})

    @property
    @abstractmethod
    def response_type(self):
        """Abstract property to be implemented by subclasses to define response type."""


class ExtractCoordinatesResponse(ExtractDataResponse):
    """Response schema for the extract_data endpoint."""

    coordinates: Coordinates = Field(
        ..., example={"east": 1.0, "north": 2.0, "page": 1, "spacial_reference_system": "LV95"}
    )

    @property
    def response_type(self):
        return "coordinates"


class ExtractElevationResponse(ExtractDataResponse):
    """Response schema for the extract_data endpoint."""

    elevation: float = Field(..., example=1.0)

    @property
    def response_type(self):
        return "elevation"


class ExtractTextResponse(ExtractDataResponse):
    """Response schema for the extract_data endpoint."""

    text: str = Field(..., example="text")

    @property
    def response_type(self):
        return "text"


class ExtractNumberResponse(ExtractDataResponse):
    """Response schema for the extract_data endpoint."""

    number: float = Field(..., example=1.0)

    @property
    def response_type(self):
        return "number"
