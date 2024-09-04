"""Schemas based on the database domain.

Each schema has a default version representing a full object,
as well as a patch version with all fields optional for patch operations.
"""

########################################################################################################################
### Creare pngs schema
########################################################################################################################

from abc import ABC, abstractmethod
from enum import Enum

import fitz
from pydantic import BaseModel, Field, constr


class PNGRequest(BaseModel):
    """Request schema for the create_pngs endpoint."""

    filename: constr(min_length=1)  # This will ensure the filename is a non-empty string.


class PNGResponse(BaseModel):
    """Response schema for the create_pngs endpoint."""

    png_urls: list[str]


########################################################################################################################
### Extract data schema
########################################################################################################################


class FormatTypes(str, Enum):
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

    def rescale(
        self, original_width: float, original_height: float, target_width: float, target_height: float
    ) -> "BoundingBox":
        """Rescale the bounding box by a factor.

        Args:
            original_width (float): The original width of the image.
            original_height (float): The original height of the image.
            target_width (float): The target width of the image.
            target_height (float): The target height of the image.

        Returns:
            BoundingBox: The rescaled bounding box.
        """
        width_factor = target_width / original_width
        height_factor = target_height / original_height

        return BoundingBox(
            x0=self.x0 * width_factor,
            y0=self.y0 * height_factor,
            x1=self.x1 * width_factor,
            y1=self.y1 * height_factor,
        )

    def to_fitz_rect(self) -> fitz.Rect:
        """Convert the bounding box to a PyMuPDF rectangle.

        Returns:
            fitz.Rect: The PyMuPDF rectangle.
        """
        return fitz.Rect(self.x0, self.y0, self.x1, self.y1)

    def load_from_fitz_rect(rect: fitz.Rect) -> "BoundingBox":
        """Load the bounding box from a PyMuPDF rectangle.

        Args:
            rect (fitz.Rect): The PyMuPDF rectangle.

        Returns:
            BoundingBox: The bounding box.
        """
        return BoundingBox(
            x0=rect.x0,
            y0=rect.y0,
            x1=rect.x1,
            y1=rect.y1,
        )


class Coordinates(BaseModel):
    """Coordinates schema."""

    east: float = Field(..., example=1.0)
    north: float = Field(..., example=2.0)
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


class NotFoundResponse(ExtractDataResponse):
    """Response schema for the not found response."""

    detail: str = Field(..., example="Resource not found.")

    @property
    def response_type(self):
        return "not_found"