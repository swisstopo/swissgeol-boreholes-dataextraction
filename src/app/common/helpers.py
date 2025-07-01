"""Helper functions for the app."""

from pathlib import Path

import numpy as np
import pymupdf

from app.common.aws import load_pdf_from_aws, load_png_from_aws


def load_pdf_page(filename: Path, page_number: int) -> pymupdf.Page:
    """Loads the page from the PDF document.

    Args:
        filename: name of the PDF file to be loaded from S3.
        page_number: number of the page (1-based)

    Returns: the PDF page as a PyMuPDF Page object
    """
    pdf_document = load_pdf_from_aws(filename)
    return pdf_document.load_page(page_number - 1)


def load_png(filename: Path, page_number: int) -> np.ndarray:
    """Loads the PNG image representation of a PDF page.

    Convert the PDF filename to a PNG filename: "10012.pdf" -> 'dataextraction/10012-1.png'

    Args:
        filename: name of the corresponding PDF file on S3
        page_number: number of the page (1-based)

    Returns: the image as a numpy array
    """
    base_filename = filename.stem
    png_filename = Path(f"{base_filename}-{page_number}.png")

    # Load the PNG file from AWS
    return load_png_from_aws(png_filename)
