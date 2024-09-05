"""This module defines the FastAPI endpoint for extracting information from PNG images."""

import fitz
from app.common.aws import load_pdf_from_aws, load_png_from_aws
from app.common.schemas import (
    BoundingBox,
    Coordinates,
    ExtractCoordinatesResponse,
    ExtractDataRequest,
    ExtractDataResponse,
    ExtractElevationResponse,
    ExtractTextResponse,
    FormatTypes,
    NotFoundResponse,
)
from stratigraphy.coordinates.coordinate_extraction import CoordinateExtractor, LV03Coordinate, LV95Coordinate
from stratigraphy.elevation.elevation_extraction import ElevationExtractor
from stratigraphy.util.extract_text import extract_text_lines_from_bbox


def extract_data(extract_data_request: ExtractDataRequest) -> ExtractDataResponse:
    """Extract information from PNG images.

    Args:
        extract_data_request (ExtractDataRequest): The request data.

    Returns:
        ExtractDataResponse: The extracted information.
    """
    # Load the PNG image
    pdf_document = load_pdf_from_aws(extract_data_request.filename)

    # Load the page from the PDF document
    pdf_page = pdf_document.load_page(extract_data_request.page_number - 1)
    pdf_page_width = pdf_page.rect.width
    pdf_page_height = pdf_page.rect.height

    # Load the PNG image the boreholes app is showing to the user
    # Convert the PDF filename to a PNG filename: "pdfs/geoquat/train/10012.pdf" -> 'pngs/geoquat/train/10012_0.png'
    # Remove the file extension and replace it with '.png'
    base_filename = extract_data_request.filename.rsplit(".", 1)[0]
    png_filename = f"{base_filename}-{extract_data_request.page_number}.png"

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

    # Extract the information based on the format type
    if extract_data_request.format == FormatTypes.COORDINATES:
        # Extract the coordinates and bounding box
        extracted_coords: ExtractCoordinatesResponse | None = extract_coordinates(
            extract_data_request, pdf_page, user_defined_bbox.to_fitz_rect()
        )

        if extracted_coords is None:
            return NotFoundResponse(
                detail="Coordinates not found.",
                bbox=extract_data_request.bbox,
            )

        # Convert the bounding box to PNG coordinates and return the response
        return ExtractCoordinatesResponse(
            bbox=extracted_coords.bbox.rescale(
                original_height=pdf_page_height,
                original_width=pdf_page_width,
                target_height=png_page_height,
                target_width=png_page_width,
            ),
            coordinates=Coordinates(
                east=extracted_coords.coordinates.east,
                north=extracted_coords.coordinates.north,
                projection=extracted_coords.coordinates.projection,
            ),
        )
    elif extract_data_request.format == FormatTypes.ELEVATION:
        extracted_elevation: ExtractElevationResponse | None = extract_elevation(
            extract_data_request, pdf_page, user_defined_bbox.to_fitz_rect()
        )

        if extracted_elevation is None:
            return NotFoundResponse(
                detail="Elevation not found.",
                bbox=extract_data_request.bbox,
            )

        # Convert the bounding box to PNG coordinates and return the response
        return ExtractElevationResponse(
            bbox=extracted_elevation.bbox.rescale(
                original_height=pdf_page_height,
                original_width=pdf_page_width,
                target_height=png_page_height,
                target_width=png_page_width,
            ),
            elevation=extracted_elevation.elevation,
        )

    elif extract_data_request.format == FormatTypes.TEXT:
        extracted_text: ExtractTextResponse = extract_text(pdf_page, user_defined_bbox.to_fitz_rect())

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
        raise NotImplementedError("Number extraction is not implemented.")
    else:
        raise ValueError("Invalid format type.")


def extract_coordinates(
    extract_data_request: ExtractDataRequest, pdf_page: fitz.Page, user_defined_bbox: fitz.Rect
) -> ExtractDataResponse | None:
    """Extract coordinates from a PNG image.

    Args:
        extract_data_request (ExtractDataRequest): The request data.
        pdf_page (fitz.Page): The PDF page.
        user_defined_bbox (fitz.Rect): The user-defined bounding box.

    Returns:
        ExtractDataResponse: The extracted coordinates.
    """

    def create_response(coord, srs):
        bbox = BoundingBox(
            x0=coord.rect.x0,
            y0=coord.rect.y0,
            x1=coord.rect.x1,
            y1=coord.rect.y1,
        )
        return ExtractCoordinatesResponse(
            bbox=bbox,
            coordinates=Coordinates(
                east=coord.east.coordinate_value,
                north=coord.north.coordinate_value,
                projection=srs,
            ),
        )

    coord_extractor = CoordinateExtractor(pdf_page)
    extracted_coord = coord_extractor.extract_coordinates_from_bbox(
        pdf_page, extract_data_request.page_number, user_defined_bbox
    )

    if isinstance(extracted_coord, LV03Coordinate):
        return create_response(extracted_coord, "LV03")

    if isinstance(extracted_coord, LV95Coordinate):
        return create_response(extracted_coord, "LV95")

    return None


def extract_elevation(
    extract_data_request: ExtractDataRequest, pdf_page: fitz.Page, user_defined_bbox: fitz.Rect
) -> ExtractDataResponse:
    """Extract the elevation from a PNG image.

    Args:
        extract_data_request (ExtractDataRequest): The request data.
        pdf_page (fitz.Page): The PDF page.
        user_defined_bbox (fitz.Rect): The user-defined bounding box.

    Returns:
        ExtractDataResponse: The extracted elevation.
    """
    elevation_extractor = ElevationExtractor(pdf_page)
    elevation = elevation_extractor.extract_elevation_from_bbox(
        pdf_page, extract_data_request.page_number, user_defined_bbox
    )

    if elevation is None:
        return NotFoundResponse(
            detail="Elevation not found.",
            bbox=extract_data_request.bbox,
        )

    bbox = BoundingBox(
        x0=elevation.rect.x0,
        y0=elevation.rect.y0,
        x1=elevation.rect.x1,
        y1=elevation.rect.y1,
    )
    return ExtractElevationResponse(
        bbox=bbox,
        elevation=elevation.elevation,
    )


def extract_text(pdf_page: fitz.Page, user_defined_bbox: fitz.Rect) -> ExtractDataResponse:
    """Extract text from a PNG image.

    Args:
        extract_data_request (ExtractDataRequest): The request data.
        pdf_page (fitz.Page): The PDF page.
        user_defined_bbox (fitz.Rect): The user-defined bounding box.

    Returns:
        ExtractDataResponse: The extracted text.
    """
    # Extract the text
    text_lines = extract_text_lines_from_bbox(pdf_page, user_defined_bbox)

    # Convert the text lines to a string
    text = ""
    text_based_bbox = fitz.Rect()
    for text_line in text_lines:
        text += text_line.text + " "
        text_based_bbox = text_based_bbox + text_line.rect

    bbox = BoundingBox.load_from_fitz_rect(text_based_bbox)
    return ExtractTextResponse(bbox=bbox, text=text)
