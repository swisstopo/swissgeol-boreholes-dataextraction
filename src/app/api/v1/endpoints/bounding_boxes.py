"""This module defines the FastAPI endpoint for extracting information from a PDF borehole document."""

from pathlib import Path

from app.common.helpers import load_pdf_page, load_png
from app.common.schemas import BoundingBox, BoundingBoxesResponse
from extraction.features.utils.text.extract_text import extract_text_lines


def bounding_boxes(filename: Path, page_number: int) -> BoundingBoxesResponse:
    """Return the bounding boxes of all words on a PDF page.

    Args:
        filename (Path): The filename of the PDF to extract bounding boxes from.
        page_number (int): The number (1-based) of the PDF page to extract bounding boxes from.

    Returns:
        BoundingBoxesResponse: The response containing a list of bounding boxes for all words on the page.
    """
    pdf_page = load_pdf_page(filename, page_number)
    pdf_page_width = pdf_page.rect.width
    pdf_page_height = pdf_page.rect.height

    # Load the PNG file from AWS
    png_page = load_png(filename, page_number)
    png_page_width = png_page.shape[1]
    png_page_height = png_page.shape[0]

    # Extract the text
    text_lines = extract_text_lines(pdf_page)

    bboxes = [
        # Convert the bounding box to PNG coordinates
        BoundingBox.load_from_pymupdf_rect(word.rect).rescale(
            original_height=pdf_page_height,
            original_width=pdf_page_width,
            target_height=png_page_height,
            target_width=png_page_width,
        )
        for line in text_lines
        for word in line.words
    ]

    return BoundingBoxesResponse(bounding_boxes=bboxes)
