"""This module defines the FastAPI endpoint for extracting information from PDF borehole document."""

import re

import pymupdf
from app.common.helpers import load_pdf_page, load_png
from app.common.schemas import (
    BoundingBox,
    Coordinates,
    ExtractCoordinatesResponse,
    ExtractDataRequest,
    ExtractDataResponse,
    ExtractNumberResponse,
    ExtractTextResponse,
    FormatTypes,
)
from borehole_extraction.extraction.metadata.coordinate_extraction import (
    CoordinateExtractor,
    LV03Coordinate,
    LV95Coordinate,
)
from borehole_extraction.extraction.util_extraction.lines.line import TextLine
from borehole_extraction.extraction.util_extraction.text.extract_text import extract_text_lines
from fastapi import HTTPException
from general_utils.file_utils import read_params
from general_utils.language_detection import detect_language_of_text

matching_params = read_params("matching_params.yml")


def extract_data(extract_data_request: ExtractDataRequest) -> ExtractDataResponse:
    """Extract information from PDF document.

    The user can specify the format type (coordinates, text,
    number) as well as the bounding box on the PNG image. The bounding box is specified in the PNG image
    coordinates (0, 0) is the top-left corner of the image.

    Args:
        extract_data_request (ExtractDataRequest): The request data with the filename, page number, bounding box,
        and format type. The page number is 1-based. The bounding box is in PNG coordinates.

    Returns:
        ExtractDataResponse: The extracted information with the bounding box where the information was found.
        In PNG coordinates.
    """
    pdf_page = load_pdf_page(extract_data_request.filename, extract_data_request.page_number)
    pdf_page_width = pdf_page.rect.width
    pdf_page_height = pdf_page.rect.height

    png_page = load_png(extract_data_request.filename, extract_data_request.page_number)
    png_page_width = png_page.shape[1]
    png_page_height = png_page.shape[0]

    # Convert the bounding box to the PDF coordinates
    user_defined_bbox = extract_data_request.bbox.rescale(
        original_height=png_page_height,
        original_width=png_page_width,
        target_height=pdf_page_height,
        target_width=pdf_page_width,
    )  # bbox in PDF coordinates

    # Select words whose middle-point is in the user-defined bbox
    text_lines = []
    for text_line in extract_text_lines(pdf_page):
        words = [
            word
            for word in text_line.words
            if user_defined_bbox.to_pymupdf_rect().contains(
                pymupdf.Point((word.rect.x0 + word.rect.x1) / 2, (word.rect.y0 + word.rect.y1) / 2)
            )
        ]
        if words:
            text_lines.append(TextLine(words))

    # Extract the information based on the format type
    if extract_data_request.format == FormatTypes.COORDINATES:
        # Detect the language of the textlines.
        # Note: language is detected only from textlines inside the user-defined bounding box, not the whole document.
        # If more confidence is needed, consider using the entire document to infer the language.
        language = detect_language_of_text(
            " ".join([text.text for text in text_lines]),
            matching_params["default_language"],
            matching_params["material_description"].keys(),
        )
        # Extract the coordinates and bounding box
        extracted_coords: ExtractCoordinatesResponse | None = extract_coordinates(
            extract_data_request, text_lines, language
        )

        # Convert the bounding box to PNG coordinates and return the response
        return ExtractCoordinatesResponse(
            bbox=extracted_coords.bbox.rescale(
                original_height=pdf_page_height,
                original_width=pdf_page_width,
                target_height=png_page_height,
                target_width=png_page_width,
            ),
            coordinates=extracted_coords.coordinates,
        )

    elif extract_data_request.format == FormatTypes.TEXT:
        extracted_text: ExtractTextResponse = extract_text(text_lines)

        # Convert the bounding box to PNG coordinates and return the response
        return ExtractTextResponse(
            bbox=extracted_text.bbox.rescale(
                original_height=pdf_page_height,
                original_width=pdf_page_width,
                target_height=png_page_height,
                target_width=png_page_width,
            ),
            text=extracted_text.text,
        )
    elif extract_data_request.format == FormatTypes.NUMBER:
        extracted_number = extract_number(text_lines)

        # Convert the bounding box to PNG coordinates and return the response
        return ExtractNumberResponse(
            bbox=extracted_number.bbox.rescale(
                original_height=pdf_page_height,
                original_width=pdf_page_width,
                target_height=png_page_height,
                target_width=png_page_width,
            ),
            number=extracted_number.number,
        )
    else:
        raise ValueError("Invalid format type.")


def extract_coordinates(
    extract_data_request: ExtractDataRequest, text_lines: list[TextLine], language: str
) -> ExtractDataResponse:
    """Extract coordinates from a collection of text lines.

    The coordinates are extracted in the Swiss coordinate system (LV03 or LV95).

    Args:
        extract_data_request (ExtractDataRequest): The request data. The page number is 1-based.
        text_lines (list[TextLine]): The text lines to extract the coordinates from.
        language (str): The language of the document.

    Returns:
        ExtractDataResponse: The extracted coordinates. The coordinates are in the Swiss coordinate
        system (LV03 or LV95). The bounding box is in PDF coordinates.
    """

    def create_response(coord_feature, srs):
        bbox = BoundingBox(
            x0=coord_feature.rect.x0,
            y0=coord_feature.rect.y0,
            x1=coord_feature.rect.x1,
            y1=coord_feature.rect.y1,
        )
        return ExtractCoordinatesResponse(
            bbox=bbox,
            coordinates=Coordinates(
                east=coord_feature.feature.east.coordinate_value,
                north=coord_feature.feature.north.coordinate_value,
                projection=srs,
            ),
        )

    coord_extractor = CoordinateExtractor(language)
    extracted_coord = coord_extractor.extract_coordinates_aggregated(text_lines, extract_data_request.page_number)

    if extracted_coord:
        extracted_coord = extracted_coord[0]  # currently we only handle one set of coordinate
        if isinstance(extracted_coord.feature, LV03Coordinate):
            return create_response(extracted_coord, "LV03")

        if isinstance(extracted_coord.feature, LV95Coordinate):
            return create_response(extracted_coord, "LV95")

    raise HTTPException(status_code=404, detail="Coordinates not found.")


def extract_text(text_lines: list[TextLine]) -> ExtractTextResponse:
    """Extract text from a collection of text lines.

    Args:
        text_lines (list[TextLine]): The text lines to extract the numbers from.

    Returns:
        ExtractTextResponse: The extracted text.
    """
    text = " ".join([line.text for line in text_lines])

    text_based_bbox = pymupdf.Rect()
    for text_line in text_lines:
        text_based_bbox = text_based_bbox.include_rect(text_line.rect)

    if text:
        bbox = BoundingBox.load_from_pymupdf_rect(text_based_bbox)
        return ExtractTextResponse(bbox=bbox, text=text)
    else:
        raise HTTPException(status_code=404, detail="Text not found.")


def extract_number(text_lines: list[TextLine]) -> ExtractNumberResponse:
    """Extract numbers from a collection of text lines.

    Args:
        text_lines (list[TextLine]): The text lines to extract the numbers from.

    Returns:
        ExtractNumberResponse: The extracted number with bbox.
    """
    for text_line in text_lines:
        number = extract_number_from_text(text_line.text)
        if number:
            bbox = BoundingBox(
                x0=text_line.rect.x0,
                y0=text_line.rect.y0,
                x1=text_line.rect.x1,
                y1=text_line.rect.y1,
            )
            return ExtractNumberResponse(bbox=bbox, number=number[0])

    raise HTTPException(status_code=404, detail="Number not found.")


def extract_number_from_text(text: str) -> list[float]:
    """Extract the number from a string.

    Args:
        text (str): The text to extract the number from. Example: "The price is 123 dollars and 45.67 cents."

    Returns:
        List[float] | None: The extracted number. Example: [123, 45.67]
    """
    # Extract all numbers (both integers and decimals)
    numbers_list = re.findall(r"\d+(?:\.\d+)?", text)

    # Convert the numbers to floats
    numbers = [float(number) for number in numbers_list]

    if len(numbers) > 1:
        raise ValueError("Multiple numbers found in the text.")

    return numbers
