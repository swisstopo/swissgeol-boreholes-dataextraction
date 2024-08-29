"""This module defines the FastAPI endpoint for converting a PDF document to PNG images."""

import os

import fitz
from app.common.aws import load_pdf_from_aws, s3_client
from app.common.config import config
from app.common.schemas import PNGResponse
from fastapi import Form, HTTPException


def create_pngs(aws_filename: str = Form(...)):
    """Convert a PDF document to PNG images.

    Args:
        aws_filename (str): The name of the PDF document in the S3 bucket. For example, "pdfs/geoquat/train/10012.pdf".

    Returns:
        PNGResponse: The URLs of the PNG images in the S3 bucket.
    """
    # Validate the filename parameter
    if not aws_filename or not isinstance(aws_filename, str):
        raise HTTPException(
            status_code=400, detail="Invalid request. 'filename' parameter is required and must be a string."
        )

    # Get the filename from the path
    filename = aws_filename.split("/")[-1].split(".")[0]
    dataset_type = aws_filename.split("/")[-2]  # The dataset name (e.g., "train")
    dataset_name = aws_filename.split("/")[-3]  # The dataset type (e.g., "geoquat")

    # Initialize the S3 client
    pdf_document = load_pdf_from_aws(aws_filename)

    png_urls = []

    # Convert each page of the PDF to PNG
    try:
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            png_filename = f"{filename}-{page_number + 1}.png"
            png_path = f"/tmp/{png_filename}"  # Local path to save the PNG
            s3_bucket_png_path = f"pngs/{dataset_name}/{dataset_type}/{png_filename}"

            pix.save(png_path)

            # Upload the PNG to S3
            s3_client.upload_file(
                filename=png_path,  # The local path to the file to upload
                bucket=config.bucket_name,  # The S3 bucket name
                key=png_filename,  # The key (name) of the file in the bucket
            )

            # Generate the S3 URL
            png_url = f"https://{config.bucket_name}.s3.amazonaws.com/{s3_bucket_png_path}"
            png_urls.append(png_url)

            # Clean up the local file
            os.remove(png_path)
    except Exception:
        raise HTTPException(status_code=500, detail="An error occurred while processing the PDF.") from None

    return PNGResponse(png_urls)
