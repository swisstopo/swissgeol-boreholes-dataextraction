"""Series of utility functions for the aws connection to get the groundwater stratigraphy."""

import io

import boto3
import fitz
import numpy as np
from app.common.config import config
from fastapi import HTTPException
from PIL import Image

# Initialize the S3 client
# AWS S3 Configuration
s3_client = boto3.client("s3")


def load_pdf_from_aws(filename: str) -> fitz.Document:
    """Load a PDF document from AWS S3.

    Args:
        filename (str): The filename of the PDF document.

    Returns:
        fitz.Document: The loaded PDF document.
    """
    # Load the PDF from the S3 object
    try:
        data = load_data_from_aws(filename, "pdfs/")
        pdf_document = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise HTTPException(
            status_code=404, detail="Failed to load PDF document. The filename is not found in the bucket."
        ) from None

    return pdf_document


def load_png_from_aws(filename: str) -> np.ndarray:
    """Load a PNG image from AWS S3.

    Args:
        filename (str): The filename of the PNG image.

    Returns:
        ndarray: The loaded PNG image.
    """
    data = load_data_from_aws(filename, "pngs/")

    # Convert the PNG data to an image using PIL
    image = Image.open(io.BytesIO(data))

    # Convert the PIL image to a NumPy array
    return np.array(image)


def load_data_from_aws(filename: str, format: str) -> bytes:
    """Load a PNG image from AWS S3.

    Args:
        filename (str): The filename of the PNG image.
        format (str): The format of the file.

    Returns:
        bytes: The loaded PNG image.
    """
    # Check if the PNG exists in S3
    try:
        png_object = s3_client.get_object(Bucket=config.bucket_name, Key=format + filename)
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Document {format + filename} not found in S3 bucket.") from None

    # Load the PNG from the S3 object
    try:
        data = png_object["Body"].read()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load data.") from None

    return data


def upload_file_to_s3(file_path: str, key: str):
    """Upload a file to S3.

    Args:
        file_path (str): The local path to the file to upload.
        key (str): The key (name) of the file in the bucket.
    """
    s3_client.upload_file(file_path, config.bucket_name, key)
