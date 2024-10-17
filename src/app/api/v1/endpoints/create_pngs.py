"""This module defines the FastAPI endpoint for converting a PDF document to PNG images."""

import os
from pathlib import Path

import fitz
from app.common.aws import load_pdf_from_aws, upload_file_to_s3
from app.common.schemas import PNGResponse
from fastapi import HTTPException


def create_pngs(aws_filename: Path) -> PNGResponse:
    """Convert a PDF document to PNG images. Please note that this function will overwrite any existing PNG files.

    Args:
        aws_filename (str): The key of the PDF document in the S3 bucket. For example, "10012.pdf".

    Returns:
        PNGResponse: The S3 keys of the PNG images in the S3 bucket.
    """
    # Check if the PDF name is valid
    if not aws_filename.suffix == ".pdf":
        raise HTTPException(status_code=400, detail="Invalid request. The filename must end with '.pdf'.")

    # Get the filename from the path
    filename = aws_filename.stem

    # Initialize the S3 client
    pdf_document = load_pdf_from_aws(aws_filename)

    s3_keys = []

    # Convert each page of the PDF to PNG
    try:
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            png_filename = f"{filename}-{page_number + 1}.png"
            png_path = f"/tmp/{png_filename}"  # Local path to save the PNG
            s3_bucket_png_path = f"dataextraction/{png_filename}"

            pix.save(png_path)

            # Upload the PNG to S3
            upload_file_to_s3(
                png_path,  # The local path to the file to upload
                s3_bucket_png_path,  # The key (name) of the file in the bucket
            )

            # Generate the S3 key
            s3_keys.append(s3_bucket_png_path)

            # Clean up the local file
            os.remove(png_path)
    except Exception:
        raise HTTPException(status_code=500, detail="An error occurred while processing the PDF.") from None

    return PNGResponse(key=s3_keys)
