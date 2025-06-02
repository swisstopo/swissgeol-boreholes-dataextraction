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

import pymupdf
from extraction.features.stratigraphy.layer.layer import Layer, LayerDepthsEntry
from extraction.features.utils.data_extractor import FeatureOnPage
from extraction.features.utils.text.textblock import MaterialDescription
from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # This allows using non-standard types like Path
        json_schema_extra={
            "example": {"filename": "geoquat/validation/1007.pdf"},
        },
    )

    @field_validator("filename", mode="before")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Ensure the filename is not empty."""
        return validate_filename(value)


class PNGResponse(BaseModel):
    """Response schema for the `create_pngs` endpoint, representing the output of PNG file creation and storage.

    This schema lists the keys (identifiers) of the created PNG files stored in an S3 bucket,
    enabling users to retrieve or reference them as needed.
    """

    keys: list[str] = Field(
        ...,
        description="""List of unique identifiers (keys) for the generated PNG files stored in the S3 bucket. Each key 
        allows access to a specific file within the bucket.""",
    )
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "keys": ["dataextraction/file1-1.png", "dataextraction/file1-2.png", "dataextraction/file1-3.png"]
            },
        }
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
    defined with the origin at the top-left of the image. Coordinates are in pixels.
    """

    x0: float = Field(
        ...,
        description="""The x-coordinate of the top-left corner of the bounding box. This value marks the 
        horizontal starting point of the box.""",
    )
    y0: float = Field(
        ...,
        description="""The y-coordinate of the top-left corner of the bounding box. This value marks the vertical 
        starting point of the box.""",
    )
    x1: float = Field(
        ...,
        description="""The x-coordinate of the bottom-right corner of the bounding box. This value marks the 
        horizontal endpoint of the box.""",
    )
    y1: float = Field(
        ...,
        description="""The y-coordinate of the bottom-right corner of the bounding box. This value marks the vertical 
        endpoint of the box.""",
    )

    model_config = ConfigDict(json_schema_extra={"example": {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0}})

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

        return self.rescale_factor(width_factor, height_factor)

    def rescale_factor(self, width_factor: float, height_factor: float) -> "BoundingBox":
        """Rescale the bounding box given a factor for each dimension.

        Args:
            width_factor (float): The ratio of the target image width by original the image width.
            height_factor (float): The ratio of the target image height by original the image height.

        Returns:
            BoundingBox: The rescaled bounding box.
        """
        return BoundingBox(
            x0=self.x0 * width_factor,
            y0=self.y0 * height_factor,
            x1=self.x1 * width_factor,
            y1=self.y1 * height_factor,
        )

    def to_pymupdf_rect(self) -> pymupdf.Rect:
        """Convert the bounding box to a PyMuPDF rectangle.

        Returns:
            pymupdf.Rect: The PyMuPDF rectangle.
        """
        return pymupdf.Rect(self.x0, self.y0, self.x1, self.y1)

    @staticmethod
    def load_from_pymupdf_rect(rect: pymupdf.Rect) -> "BoundingBox":
        """Load the bounding box from a PyMuPDF rectangle.

        Args:
            rect (pymupdf.Rect): The PyMuPDF rectangle.

        Returns:
            BoundingBox: The bounding box.
        """
        return BoundingBox(
            x0=rect.x0,
            y0=rect.y0,
            x1=rect.x1,
            y1=rect.y1,
        )


class BoundingBoxWithPage(BoundingBox):
    """Extends BoundingBox by adding page number metadata."""

    page_number: int = Field(
        ...,
        description="The page number in the PDF where this bounding box is located (1-based index).",
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"page_number": 2, "x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0}}
    )

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Page number must be a positive integer")
        return v

    def rescale(
        self, original_width: float, original_height: float, target_width: float, target_height: float
    ) -> "BoundingBoxWithPage":
        """Rescale the bounding box by a factor given the original and target dimensions.

        Args:
            original_width (float): The original width of the image.
            original_height (float): The original height of the image.
            target_width (float): The target width of the image.
            target_height (float): The target height of the image.

        Returns:
            BoundingBoxWithPage: The rescaled bounding box.
        """
        width_factor = target_width / original_width
        height_factor = target_height / original_height

        return self.rescale_factor(width_factor, height_factor)

    def rescale_factor(self, width_factor: float, height_factor: float) -> "BoundingBoxWithPage":
        """Rescale the bounding box given a factor for each dimension.

        Args:
            width_factor (float): The ratio of the target image width by original the image width.
            height_factor (float): The ratio of the target image height by original the image height.

        Returns:
            BoundingBoxWithPage: The rescaled bounding box.
        """
        return BoundingBoxWithPage(
            x0=self.x0 * width_factor,
            y0=self.y0 * height_factor,
            x1=self.x1 * width_factor,
            y1=self.y1 * height_factor,
            page_number=self.page_number,
        )


class Coordinates(BaseModel):
    """Coordinates schema for representing geographical data points.

    This schema defines the format for specifying location data using east/north coordinates
    along with the projection system used.
    """

    east: float = Field(
        ..., description="""Easting coordinate. The value should be in the units of the specified projection system."""
    )
    north: float = Field(
        ...,
        description="""Northing coordinate. The value should be in the units of the specified projection system.""",
    )
    projection: str = Field(
        ...,
        description="""Projection system used to reference the coordinates. This defines the coordinate reference
        system, such as 'LV95' for Swiss coordinate systems.""",
    )

    model_config = ConfigDict(json_schema_extra={"example": {"east": 1.0, "north": 2.0, "projection": "LV95"}})


class BoundingBoxesRequest(ABC, BaseModel):
    """Request schema for the `bounding_boxes` endpoint.

    ### Fields
    Each field below includes inline examples to aid users in creating requests. See `json_schema_extra`
    for a complete example.

    **Attributes:**
    - **filename** (`Path`): Path to the PDF file. _Example_: `"document.pdf"`
    - **page_number** (`int`): Target page for data extraction. This is a 1-based index. _Example_: `1`

    ### Validation
    Custom validators ensure data integrity:
    - **Filename Validator:** Ensures filename is not empty.
    - **Page Number Validator:** Confirms page number is positive.
    """

    filename: Path = Field(
        ...,
        description="""Path to the input PDF document file that contains the data to be extracted. This should be
        a valid file path, and the file should be accessible to the API.""",
    )

    page_number: int = Field(
        ...,
        description="""Page number within the document where the extraction is to be performed. This is a 1-based 
        index (e.g., 1 for the first page), applicable for multi-page files like PDFs.""",
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"filename": "geoquat/validation/1007.pdf", "page_number": 1}}
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


class BoundingBoxesResponse(BaseModel):
    """Response schema for the `bounding_boxes` endpoint, representing the bounding boxes of words on the page."""

    bounding_boxes: list[BoundingBox] = Field(
        ..., description="""List of bounding boxes for all words that are found on the requested page."""
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bounding_boxes": [
                    {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0},
                    {"x0": 150.0, "y0": 20.0, "x1": 220.0, "y1": 40.0},
                ]
            }
        }
    )


class ExtractDataRequest(ABC, BaseModel):
    """Request schema for the `extract_data` endpoint.

    ** Requirements:**
    Before using this schema, ensure that the PDF file has been processed by the create_pngs endpoint first.

    **Coordinate Systems:**
    - **PNG coordinates:** Pixels are measured from the top-left corner (0, 0), where x increases rightward
    and y downward.

    ### Fields
    Each field below includes inline examples to aid users in creating requests. See `json_schema_extra`
    for a complete example.

    **Attributes:**
    - **filename** (`Path`): Path to the PDF file. _Example_: `"document.pdf"`
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

    The bounding box should be provided in PNG coordinates.

    Each field in the Pydantic model can have an example parameter, which provides an inline
    example for that specific field.
    """

    filename: Path = Field(
        ...,
        description="""Path to the input PDF document file that contains the data to be extracted. This should be
        a valid file path, and the file should be accessible to the API.""",
    )
    page_number: int = Field(
        ...,
        description="""Page number within the document where the extraction is to be performed. This is a 1-based 
        index (e.g., 1 for the first page), applicable for multi-page files like PDFs.""",
    )
    bbox: BoundingBox = Field(
        ...,
        description="""Bounding box defining the area for data extraction within the PNG version of the specified 
        PDF file. The box is specified in pixels with the top-left as the origin (0,0), where x increases to the 
        right and y increases downward. This box should be provided in PNG coordinates, and any 
        transformations to PDF coordinates are managed internally.
        """,
    )
    format: FormatTypes = Field(
        ...,
        description="""Specifies the desired format for extracted data, allowing for options like `coordinates` or 
        other defined `FormatTypes` values. This dictates the structure of the output returned by the API.""",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "geoquat/validation/1007.pdf",
                "page_number": 1,
                "bbox": {"x0": 0.0, "y0": 0.0, "x1": 200.0, "y1": 200.0},
                "format": FormatTypes.COORDINATES.value,  # Adjust as needed
            }
        }
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


class ExtractDataResponse(ABC, BaseModel):
    """Base response schema for the `extract_data` endpoint, representing the extracted data's bounding box.

    This abstract base class provides a bounding box field for data localization and an abstract property
    `response_type` to be implemented by subclasses, indicating the type of extracted content.
    """

    bbox: BoundingBox = Field(
        ...,
        description="""Bounding box coordinates that define the area within the document where data was extracted.
        The box is specified in PNG coordinates, with the origin at the top-left corner (0,0).""",
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"bbox": {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0}}}
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
        description="""Geographical coordinates extracted from the document, including east and north values, 
        and projection type.""",
    )
    model_config = ConfigDict(
        json_schema_extra={"example": {"coordinates": {"east": 1.0, "north": 2.0, "projection": "LV95"}}}
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
        description="""Text content extracted from the specified bounding box within the document.""",
    )
    model_config = ConfigDict(json_schema_extra={"example": {"text": "text"}})

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
        description="""Numeric value extracted from the specified bounding box within the document, representing a
        measurement or quantitative data.""",
    )

    model_config = ConfigDict(json_schema_extra={"example": {"number": 1.0}})

    @property
    def response_type(self):
        return "number"


class LayerMaterialDescriptionSchema(BaseModel):
    """Schema for representing a material description within a borehole layer.

    A material description consists of a text string (e.g., "Kies", "Ton") and one or
    more bounding boxes across one or more pages that visually localize the description
    within the source document.
    """

    text: str
    bounding_boxes: list[BoundingBoxWithPage]

    @classmethod
    def from_prediction(
        cls, prediction: FeatureOnPage[MaterialDescription], pdf_img_scalings: list[tuple[float]]
    ) -> "LayerMaterialDescriptionSchema":
        return cls(
            text=prediction.feature.text,
            bounding_boxes=[
                BoundingBoxWithPage(
                    page_number=line_feature.page,
                    x0=line_feature.rect.x0,
                    y0=line_feature.rect.y0,
                    x1=line_feature.rect.x1,
                    y1=line_feature.rect.y1,
                ).rescale_factor(*pdf_img_scalings[line_feature.page - 1])
                for line_feature in prediction.feature.lines
            ],
        )


class LayerDepthSchema(BaseModel):
    """Schema for representing a depth marker within a borehole layer.

    A depth marker includes a depth value (in meters) and a list of bounding boxes
    that indicate where this depth is mentioned or displayed within the PDF.
    """

    depth: float
    bounding_boxes: list[BoundingBoxWithPage]

    @classmethod
    def from_prediction(
        cls, prediction: LayerDepthsEntry, pdf_img_scalings: list[tuple[float]], fallback_page: int
    ) -> "LayerDepthSchema":
        page_scalings = pdf_img_scalings[fallback_page - 1]
        return cls(
            depth=prediction.value,
            bounding_boxes=[
                BoundingBoxWithPage(
                    page_number=fallback_page,  # page is taken from material_description as a fallback
                    x0=prediction.rect.x0,
                    y0=prediction.rect.y0,
                    x1=prediction.rect.x1,
                    y1=prediction.rect.y1,
                ).rescale_factor(*page_scalings)
            ],
        )


class BoreholeLayerSchema(BaseModel):
    """Schema for a single stratigraphic layer in a borehole.

    Each layer may consist of a material description, a start depth, and an end depth.
    Any of these components may be absent if not detected during extraction.
    """

    material_description: LayerMaterialDescriptionSchema | None
    start: LayerDepthSchema | None
    end: LayerDepthSchema | None

    @classmethod
    def from_prediction(cls, prediction: Layer, pdf_img_scalings: list[tuple[float]]) -> "BoreholeLayerSchema":
        material_descr = prediction.material_description
        material_descr = (
            LayerMaterialDescriptionSchema.from_prediction(material_descr, pdf_img_scalings)
            if material_descr
            else None
        )

        fallback_page = material_descr.bounding_boxes[0].page_number if material_descr else 1
        start = (
            LayerDepthSchema.from_prediction(prediction.depths.start, pdf_img_scalings, fallback_page)
            if prediction.depths and prediction.depths.start
            else None
        )
        end = (
            LayerDepthSchema.from_prediction(prediction.depths.end, pdf_img_scalings, fallback_page)
            if prediction.depths and prediction.depths.end
            else None
        )

        return cls(
            material_description=material_descr,
            start=start,
            end=end,
        )


class BoreholeExtractionSchema(BaseModel):
    """Schema for representing an extracted borehole and its associated layers.

    Each borehole includes the list of pages it spans (`page_numbers`) and a list of
    extracted stratigraphic layers (`layers`). This allows downstream consumers to
    map layers to visual representations in the PDF and construct full profiles.
    """

    page_numbers: list[int]
    layers: list[BoreholeLayerSchema]


class ExtractStratigraphyRequest(ABC, BaseModel):
    """Request schema for the `extract_stratigraphy` endpoint.

    This endpoint processes a PDF and extracts all borehole stratigraphy information
    without needing page number or bounding box input.

    ### Fields

    **Attributes:**
    - **filename** (`Path`): Path to the PDF file. _Example_: `"geoquat/validation/1007.pdf"`

    ### Validation
    - **Filename Validator:** Ensures filename is not empty.
    """

    filename: Path = Field(
        ...,
        description="""Path to the input PDF document that contains borehole stratigraphy. 
        This must be a valid file path, and the file should be accessible by the API.""",
    )

    model_config = ConfigDict(json_schema_extra={"example": {"filename": "geoquat/validation/1007.pdf"}})

    @field_validator("filename", mode="before")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Ensure the filename is not empty."""
        return validate_filename(value)


class ExtractStratigraphyResponse(BaseModel):
    """Response schema for the `extract_stratigraphy` endpoint.

    Returns structured borehole information extracted from the full PDF document.

    ### Fields
    - **boreholes** (`List[BoreholeExtraction]`): List of extracted borehole entries.
    """

    boreholes: list[BoreholeExtractionSchema] = Field(
        ...,
        description="List of all boreholes extracted from the document, including stratigraphy layers and metadata.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "boreholes": [
                    {
                        "page_numbers": [2, 3],
                        "layers": [
                            {
                                "material_description": {
                                    "text": "Kies",
                                    "bounding_boxes": [
                                        {"page_number": 2, "x0": 100.0, "y0": 50.0, "x1": 200.0, "y1": 65.0}
                                    ],
                                },
                                "start": {
                                    "depth": 0.0,
                                    "bounding_boxes": [
                                        {"page_number": 2, "x0": 90.0, "y0": 45.0, "x1": 120.0, "y1": 60.0}
                                    ],
                                },
                                "end": {
                                    "depth": 1.2,
                                    "bounding_boxes": [
                                        {"page_number": 2, "x0": 90.0, "y0": 70.0, "x1": 120.0, "y1": 85.0}
                                    ],
                                },
                            },
                            {
                                "material_description": {
                                    "text": "Ton",
                                    "bounding_boxes": [
                                        {"page_number": 3, "x0": 100.0, "y0": 50.0, "x1": 200.0, "y1": 65.0}
                                    ],
                                },
                                "start": {
                                    "depth": 0.0,
                                    "bounding_boxes": [
                                        {"page_number": 3, "x0": 90.0, "y0": 45.0, "x1": 120.0, "y1": 60.0}
                                    ],
                                },
                                "end": {
                                    "depth": 1.2,
                                    "bounding_boxes": [
                                        {"page_number": 3, "x0": 90.0, "y0": 70.0, "x1": 120.0, "y1": 85.0}
                                    ],
                                },
                            },
                        ],
                    }
                ]
            }
        }
    )
