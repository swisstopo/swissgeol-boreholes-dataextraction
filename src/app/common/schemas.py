"""Schemas based on the database domain.

Each schema has a default version representing a full object,
as well as a patch version with all fields optional for patch operations.
"""

########################################################################################################################
### Create pngs schema
########################################################################################################################

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import fitz
from pydantic import BaseModel, Field, field_validator


def validate_filename(value: str) -> str:
    """Ensure the filename is not empty.

    Args:
        value (str): The filename to validate.

    Returns:
        str: The validated filename.

    Raises:
        ValueError: If the filename is empty
    """
    if value == "":
        raise ValueError("Filename must not be empty.")
    return value


class PNGRequest(BaseModel):
    """Request schema for the create_pngs endpoint."""

    filename: Path  # This will ensure the filename is a Path object

    @field_validator("filename", mode="before")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Ensure the filename is not empty."""
        return validate_filename(value)

    class Config:
        """Make to allow using non-standard types like Path."""

        arbitrary_types_allowed: bool = True  # This allows using non-standard types like Path


class PNGResponse(BaseModel):
    """Response schema for the `create_pngs` endpoint, representing the output of PNG file creation and storage.

    This schema lists the keys (identifiers) of the created PNG files stored in an S3 bucket,
    enabling users to retrieve or reference them as needed.
    """

    keys: list[str] = Field(
        ...,
        description="""
            List of unique identifiers (keys) for the generated PNG files stored in the S3 bucket. Each key allows 
            access to a specific file within the bucket.
        """,
        example=[
            "dataextraction/file1-1.png",
            "dataextraction/file1-2.png",
            "dataextraction/file2-1.png",
            "dataextraction/file3-1.png",
        ],
    )


########################################################################################################################
### Extract data schema
########################################################################################################################


class FormatTypes(str, Enum):
    """Enum for the format types."""

    TEXT = "text"
    NUMBER = "number"
    COORDINATES = "coordinates"


class BoundingBox(BaseModel):
    """Bounding box schema for defining a rectangular area within an image.

    This schema represents the coordinates of the box’s corners, which can be used
    to specify an area of interest in image processing tasks. Coordinates are
    defined with the origin at the top-left of the image.
    """

    x0: float = Field(
        ...,
        description="""
            The x-coordinate of the top-left corner of the bounding box. This value marks the horizontal starting 
            point of the box.
        """,
        example=0.0,
    )
    y0: float = Field(
        ...,
        description=""""
            The y-coordinate of the top-left corner of the bounding box. This value marks the vertical starting 
            point of the box.
        """,
        example=0.0,
    )
    x1: float = Field(
        ...,
        description="""
            The x-coordinate of the bottom-right corner of the bounding box. This value marks the horizontal 
            endpoint of the box.
        """,
        example=100.0,
    )
    y1: float = Field(
        ...,
        description="""
            The y-coordinate of the bottom-right corner of the bounding box. This value marks the vertical 
            endpoint of the box.
        """,
        example=100.0,
    )

    @field_validator("x0", "y0", "x1", "y1")
    @classmethod
    def bbox_corners_must_be_positive(cls, v: int) -> int:
        """Validate that the edges of the bounding box are positive."""
        if v < 0.0:
            raise ValueError("Bounding box coordinates must be positive")
        return v

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

    @staticmethod
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
    """Coordinates schema for representing geographical data points.

    This schema defines the format for specifying location data using east/north coordinates
    along with the projection system used for geo-referencing.
    """

    east: float = Field(
        ...,
        description="""
            Easting coordinate, representing the horizontal position of the point. The value should be in the 
            units of the specified projection system.
        """,
        example=1.0,
    )
    north: float = Field(
        ...,
        description="""
            Northing coordinate, representing the vertical position of the point. The value should be in the 
            units of the specified projection system.
        """,
        example=2.0,
    )
    projection: str = Field(
        ...,
        description="""
            Projection system used to reference the coordinates. This defines the coordinate reference system, 
            such as 'LV95' for Swiss coordinate systems.
        """,
        example="LV95",
    )


class ExtractDataRequest(ABC, BaseModel):
    """Request schema for the `extract_data` endpoint.

    **Coordinate Systems:**
    - **PNG coordinates:** Pixels are measured from the top-left corner (0, 0), where x increases rightward
    and y downward.
    - **PDF coordinates:** Also measured from the top-left corner (0, 0), though any transformations between
    PDF and PNG coordinates are managed internally by the `BoundingBox.rescale` method.

    ### Fields
    Each field below includes inline examples to aid users in creating requests. See `json_schema_extra`
    for a complete example.

    **Attributes:**
    - **filename** (`Path`): Path to the file (PNG or PDF). _Example_: `"document.png"`
    - **page_number** (`int`): Target page for data extraction. This is a 1-based index. _Example_: `1`
    - **bbox** (`BoundingBox`): Bounding box for the extraction area, in PNG coordinates. Origin is the
    top-left, with x increasing rightward and y increasing downward.
        - Example format: `{"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0}`
    - **format** (`FormatTypes`): Specifies the expected format for extracted data, e.g., `"coordinates"`.

    ### Validation
    Custom validators ensure data integrity:
    - **Filename Validator:** Ensures filename is not empty.
    - **Page Number Validator:** Confirms page number is positive.
    - **Format Validator:** Checks format is valid as per `FormatTypes`.


    The bounding box should be provided in PNG coordinates. Any necessary coordinate transformations between PNG
    and PDF are handled internally using the BoundingBox.rescale method.

    Each field in the Pydantic model can have an example parameter, which provides an inline
    example for that specific field.
    """

    filename: Path = Field(
        ...,
        description="""
            Path to the input document file (PNG or PDF) that contains the data to be extracted. This should be
            a valid file path, and the file should be accessible to the API.
        """,
        example=Path("document.png"),
    )
    page_number: int = Field(
        ...,
        description="""
            Page number within the document where the extraction is to be performed. This is a 1-based 
            index (e.g., 1 for the first page), applicable for multi-page files like PDFs.
        """,
        example=1,
    )
    bbox: BoundingBox = Field(
        ...,
        description="""
            Bounding box defining the area for data extraction within the PNG image. The box is specified in 
            pixels with the top-left as the origin (0,0), where x increases to the right and y increases 
            downward. This box should be provided in PNG coordinates, and any transformations to PDF
            coordinates are managed internally.
        """,
        example={"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0},
    )
    format: FormatTypes = Field(
        ...,
        description="""
            Specifies the desired format for extracted data, allowing for options like `coordinates` or other
            defined `FormatTypes` values. This dictates the structure of the output returned by the API.
        """,
        example=FormatTypes.COORDINATES.value,
    )

    @field_validator("filename", mode="before")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Ensure the filename is not empty."""
        return validate_filename(value)

    @field_validator("page_number")
    @classmethod
    def page_number_must_be_positive(cls, v: int) -> int:
        """Validate that the page number is positive."""
        if v <= 0:
            raise ValueError("Page number must be a positive integer")
        return v

    @field_validator("format")
    @classmethod
    def format_must_be_valid(cls, v: FormatTypes) -> FormatTypes:
        """Validate that the format is valid."""
        if v not in FormatTypes:
            raise ValueError(f"Invalid format type: {v}")
        return v

    class Config:
        """Make it possible to define an example for the entire request model in the Swagger UI.

        The schema_extra attribute inside the Config class allows you to define a complete
        example for the entire request model.
        """

        json_schema_extra = {
            "example": {
                "filename": "10012.pdf",
                "page_number": 1,
                "bbox": {"x0": 0.0, "y0": 0.0, "x1": 200.0, "y1": 200.0},
                "format": "coordinates",  # Adjust this to match your actual FormatTypes
            }
        }


class ExtractDataResponse(ABC, BaseModel):
    """Base response schema for the `extract_data` endpoint, representing the extracted data's bounding box.

    This abstract base class provides a bounding box field for data localization and an abstract property
    `response_type` to be implemented by subclasses, indicating the type of extracted content.
    """

    bbox: BoundingBox = Field(
        ...,
        description="""
            Bounding box coordinates that define the area within the document where data was extracted. The box
            is specified in PNG coordinates, with the origin at the top-left corner (0,0).
        """,
        example={"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0},
    )

    @property
    @abstractmethod
    def response_type(self):
        """Abstract property to be implemented by subclasses to define the type of response content."""


class ExtractCoordinatesResponse(ExtractDataResponse):
    """Response schema for the `extract_data` endpoint when returning geographic coordinates.

    This schema includes a `coordinates` field with east/north values and projection information.
    """

    coordinates: Coordinates = Field(
        ...,
        description="""
            Geographical coordinates extracted from the document, including east, north values, page number,
            and projection type.
        """,
        example={"east": 1.0, "north": 2.0, "page": 1, "projection": "LV95"},
    )

    @property
    def response_type(self):
        return "coordinates"


class ExtractTextResponse(ExtractDataResponse):
    """Response schema for the `extract_data` endpoint when returning extracted text content.

    This schema includes a `text` field with the extracted textual content from the specified bounding box.
    """

    text: str = Field(
        ...,
        description="""
            Text content extracted from the specified bounding box within the document.
        """,
        example="text",
    )

    @property
    def response_type(self):
        return "text"


class ExtractNumberResponse(ExtractDataResponse):
    """Response schema for the `extract_data` endpoint when returning numerical data.

    This schema includes a `number` field for extracted numeric content, such as measurements or other
    quantitative data.
    """

    number: float = Field(
        ...,
        description="""
            Numeric value extracted from the specified bounding box within the document, representing a 
            measurement or quantitative data.
        """,
        example=1.0,
    )

    @property
    def response_type(self):
        return "number"
