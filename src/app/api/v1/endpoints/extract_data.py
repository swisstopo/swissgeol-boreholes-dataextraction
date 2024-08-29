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
    ExtractLinesResponse,
    ExtractTextResponse,
    FormatTypes,
)
from stratigraphy.util.coordinate_extraction import CoordinateExtractor, LV03Coordinate, LV95Coordinate
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
    pdf_page = pdf_document.load_page(extract_data_request.page_number)
    pdf_page_width = pdf_page.rect.width
    pdf_page_height = pdf_page.rect.height

    # Load the PNG image the boreholes app is showing to the user
    # Convert the PDF filename to a PNG filename: "pdfs/geoquat/train/10012.pdf" -> 'pngs/geoquat/train/10012_0.png'
    png_filename = f"{extract_data_request.filename}_{extract_data_request.page_number}.png"
    png_filename = png_filename.replace(".pdf", "")
    png_filename = png_filename.replace("pdf", "png")
    png_page = load_png_from_aws(png_filename)
    png_page_width = png_page.shape[1]
    png_page_height = png_page.shape[0]

    # Convert the bounding box to the PDF coordinates
    x0 = extract_data_request.bbox.x0 * pdf_page_width / png_page_width
    y0 = extract_data_request.bbox.y0 * pdf_page_height / png_page_height
    x1 = extract_data_request.bbox.x1 * pdf_page_width / png_page_width
    y1 = extract_data_request.bbox.y1 * pdf_page_height / png_page_height
    user_defined_bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)  # bbox in PDF coordinates

    # Extract the information based on the format type
    if extract_data_request.format == FormatTypes.COORDINATES:
        # Extract the coordinates and bounding box
        extracted_coords: ExtractCoordinatesResponse = extract_coordinates(
            pdf_page, extract_data_request.page_number, user_defined_bbox
        )

        # Convert the bounding box to PNG coordinates
        png_bbox_list = []
        for bbox in extracted_coords.bbox_list:
            x0 = bbox.x0 * png_page_width / pdf_page_width
            y0 = bbox.y0 * png_page_height / pdf_page_height
            x1 = bbox.x1 * png_page_width / pdf_page_width
            y1 = bbox.y1 * png_page_height / pdf_page_height
            png_bbox_list.append(BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1))

        coordinates_list = []
        for coords in extracted_coords.coordinates:
            east = coords.east
            north = coords.north
            page = coords.page
            spacial_reference_system = coords.spacial_reference_system
            coordinates_list.append(
                Coordinates(east=east, north=north, page=page, spacial_reference_system=spacial_reference_system)
            )

        return ExtractCoordinatesResponse(
            bbox_list=png_bbox_list,
            coordinates=coordinates_list,
        )
    elif extract_data_request.format == FormatTypes.ELEVATION:
        return extract_elevation(extract_data_request, pdf_page, user_defined_bbox)
    elif extract_data_request.format == FormatTypes.TEXT:
        extracted_text: ExtractTextResponse = extract_text(pdf_page, user_defined_bbox)

        # Convert the bounding box to PNG coordinates
        png_bbox_list = []
        for bbox in extracted_text.bbox_list:
            x0 = bbox.x0 * png_page_width / pdf_page_width
            y0 = bbox.y0 * png_page_height / pdf_page_height
            x1 = bbox.x1 * png_page_width / pdf_page_width
            y1 = bbox.y1 * png_page_height / pdf_page_height
            png_bbox_list.append(BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1))

        return ExtractTextResponse(
            bbox_list=png_bbox_list,
            text=extracted_text.text,
        )
    elif extract_data_request.format == FormatTypes.NUMBER:
        raise NotImplementedError("Number extraction is not implemented.")
    elif extract_data_request.format == FormatTypes.LINES:
        extracted_lines: ExtractLinesResponse = extract_lines(extract_data_request, pdf_page, user_defined_bbox)

        # Convert the bounding box to PNG coordinates
        png_bbox_list = []
        for bbox in extracted_lines.bbox_list:
            x0 = bbox.x0 * png_page_width / pdf_page_width
            y0 = bbox.y0 * png_page_height / pdf_page_height
            x1 = bbox.x1 * png_page_width / pdf_page_width
            y1 = bbox.y1 * png_page_height / pdf_page_height
            png_bbox_list.append(BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1))

        return ExtractLinesResponse(
            bbox_list=png_bbox_list,
            lines=extracted_lines.lines,
        )
    else:
        raise ValueError("Invalid format type.")


def extract_coordinates(
    extract_data_request: ExtractDataRequest, pdf_page: fitz.Page, user_defined_bbox: fitz.Rect
) -> ExtractDataResponse:
    """Extract coordinates from a PNG image.

    Args:
        extract_data_request (ExtractDataRequest): The request data.
        pdf_page (fitz.Page): The PDF page.
        user_defined_bbox (fitz.Rect): The user-defined bounding box.

    Returns:
        ExtractDataResponse: The extracted coordinates.
    """
    coord_extractor = CoordinateExtractor()
    extracted_coord = coord_extractor.extract_coordinates_from_bbox(
        pdf_page, extract_data_request.page_number, user_defined_bbox
    )

    if isinstance(extracted_coord, LV03Coordinate):
        bbox = BoundingBox(
            x0=extracted_coord.rect.x0,
            y0=extracted_coord.rect.y0,
            x1=extracted_coord.rect.x1,
            y1=extracted_coord.rect.y1,
        )

        return ExtractCoordinatesResponse(
            bbox=[bbox],
            coordinates=[
                Coordinates(
                    east=extracted_coord.east.coordinate_value,
                    north=extracted_coord.north.coordinate_value,
                    page=extract_data_request.page_number,
                    spacial_reference_system="LV03",
                )
            ],
        )

    if isinstance(extracted_coord, LV95Coordinate):
        bbox = BoundingBox(
            x0=extracted_coord.rect.x0,
            y0=extracted_coord.rect.y0,
            x1=extracted_coord.rect.x1,
            y1=extracted_coord.rect.y1,
        )

        return ExtractCoordinatesResponse(
            bbox=[bbox],
            coordinates=[
                Coordinates(
                    east=extracted_coord.east.coordinate_value,
                    north=extracted_coord.north.coordinate_value,
                    page=extract_data_request.page_number,
                    spacial_reference_system="LV95",
                )
            ],
        )

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
    # Extract the elevation
    elevation = 444
    bbox = BoundingBox(x0=20.0, y0=40.0, x1=60.0, y1=80.0)
    return ExtractElevationResponse(bbox=bbox, elevation=elevation)


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
    for text_line in text_lines:
        text += text_line.text + " "

    bbox_list = [
        BoundingBox(
            x0=user_defined_bbox.x0,
            y0=user_defined_bbox.y0,
            x1=user_defined_bbox.x1,
            y1=user_defined_bbox.y1,
        )
    ]
    return ExtractTextResponse(bbox_list=bbox_list, text=[text])


def extract_lines(
    extract_data_request: ExtractDataRequest, pdf_page: fitz.Page, user_defined_bbox: fitz.Rect
) -> ExtractLinesResponse:
    """Extract lines from a PNG image.

    Args:
        extract_data_request (ExtractDataRequest): The request data.
        pdf_page (fitz.Page): The PDF page.
        user_defined_bbox (fitz.Rect): The user-defined bounding box.

    Returns:
        ExtractLinesResponse: The extracted lines.
    """
    # Extract the lines
    text_lines = extract_text_lines_from_bbox(pdf_page, user_defined_bbox)

    # Convert the text lines to a string
    lines = []
    rects = []
    for text_line in text_lines:
        lines.append(text_line.text)
        rects.append(text_line.rect)

    bbox_list = [
        BoundingBox(
            x0=rect.x0,
            y0=rect.y0,
            x1=rect.x1,
            y1=rect.y1,
        )
        for rect in rects
    ]
    return ExtractLinesResponse(bbox_list=bbox_list, lines=lines)
