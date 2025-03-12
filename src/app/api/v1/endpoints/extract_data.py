"""This module defines the FastAPI endpoint for extracting information from PDF borehole document."""

import re
from pathlib import Path

import fitz
from app.common.aws import load_pdf_from_aws, load_png_from_aws
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
from fastapi import HTTPException
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.metadata.coordinate_extraction import CoordinateExtractor, LV03Coordinate, LV95Coordinate
from stratigraphy.text.extract_text import extract_text_lines_from_bbox


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
    # Load the PNG image
    pdf_document = load_pdf_from_aws(extract_data_request.filename)

    # Load the page from the PDF document
    pdf_page = pdf_document.load_page(extract_data_request.page_number - 1)
    pdf_page_width = pdf_page.rect.width
    pdf_page_height = pdf_page.rect.height

    # Load the PNG image the boreholes app is showing to the user
    # Convert the PDF filename to a PNG filename: "10012.pdf" -> 'dataextraction/10012-1.png'
    # Remove the file extension and replace it with '.png'
    base_filename = extract_data_request.filename.stem
    png_filename = Path(f"{base_filename}-{extract_data_request.page_number}.png")

    # Load the PNG file from AWS
    png_page = load_png_from_aws(png_filename)
    png_page_width = png_page.shape[1]
    png_page_height = png_page.shape[0]

    # Convert the bounding box to the PDF coordinates
    user_defined_bbox = extract_data_request.bbox.rescale(
        original_height=png_page_height,
        original_width=png_page_width,
        target_height=pdf_page_height,
        target_width=pdf_page_width,
    )  # bbox in PDF coordinates

    # If the PDF page is rotated, rotate the bounding box
    user_defined_rect_with_rotation = user_defined_bbox.to_fitz_rect() * pdf_page.derotation_matrix

    # Extract the information based on the format type
    if extract_data_request.format == FormatTypes.COORDINATES:
        # Extract the coordinates and bounding box
        extracted_coords: ExtractCoordinatesResponse | None = extract_coordinates(
            extract_data_request, pdf_page, user_defined_rect_with_rotation
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
        extracted_text: ExtractTextResponse = extract_text(pdf_page, user_defined_rect_with_rotation)

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
        extracted_number = extract_number(pdf_page, user_defined_rect_with_rotation)

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
    extract_data_request: ExtractDataRequest, pdf_page: fitz.Page, user_defined_bbox: fitz.Rect
) -> ExtractDataResponse:
    """Extract coordinates from a PDF document.

    The coordinates are extracted from the user-defined bounding box. The coordinates are extracted in the
    Swiss coordinate system (LV03 or LV95). The user_defined_bbox is in PDF coordinates.

    Args:
        extract_data_request (ExtractDataRequest): The request data. The page number is 1-based.
        pdf_page (fitz.Page): The PDF page.
        user_defined_bbox (fitz.Rect): The user-defined bounding box. The bounding box is in PDF coordinates.

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

    coord_extractor = CoordinateExtractor()
    extracted_coord = coord_extractor.extract_coordinates_from_bbox(
        pdf_page, extract_data_request.page_number, user_defined_bbox
    )
    if extracted_coord:
        extracted_coord = extracted_coord[0]  # currently we only handles one set of coordinate

    if isinstance(extracted_coord, FeatureOnPage):
        if isinstance(extracted_coord.feature, LV03Coordinate):
            return create_response(extracted_coord, "LV03")

        if isinstance(extracted_coord.feature, LV95Coordinate):
            return create_response(extracted_coord, "LV95")

    raise HTTPException(status_code=404, detail="Coordinates not found.")


def extract_text(pdf_page: fitz.Page, user_defined_bbox: fitz.Rect) -> ExtractDataResponse:
    """Extract text from a PDF Document. The text is extracted from the user-defined bounding box.

    Args:
        pdf_page (fitz.Page): The PDF page.
        user_defined_bbox (fitz.Rect): The user-defined bounding box. The bounding box is in PDF coordinates.

    Returns:
        ExtractDataResponse: The extracted text.
    """
    # Extract the text
    text_lines = extract_text_lines_from_bbox(pdf_page, user_defined_bbox)

    # Convert the text lines to a string
    text = " ".join([text_line.text for text_line in text_lines])

    text_based_bbox = fitz.Rect()
    for text_line in text_lines:
        text_based_bbox = text_based_bbox.include_rect(text_line.rect)

    if text:
        bbox = BoundingBox.load_from_fitz_rect(text_based_bbox)
        return ExtractTextResponse(bbox=bbox, text=text)
    else:
        raise HTTPException(status_code=404, detail="Text not found.")


def extract_number(pdf_page: fitz.Page, user_defined_bbox: fitz.Rect) -> ExtractNumberResponse:
    """Extract numbers from a PDF document. The numbers are extracted from the user-defined bounding box.

    Args:
        pdf_page (fitz.Page): The PDF page.
        user_defined_bbox (fitz.Rect): The user-defined bounding box. The bounding box is in PDF coordinates.

    Returns:
        ExtractDataResponse: The extracted number with bbox.
    """
    # Extract the text
    text_lines = extract_text_lines_from_bbox(pdf_page, user_defined_bbox)

    # Extract the number
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
