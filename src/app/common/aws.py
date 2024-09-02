"""Series of utility functions for the aws connection to get the groundwater stratigraphy."""

import io

import fitz
import numpy as np
from app.common.config import config
from boto3 import client
from fastapi import HTTPException
from PIL import Image

# Initialize the S3 client
# AWS S3 Configuration
s3_client = client("s3")


def load_pdf_from_aws(filename: str) -> fitz.Document:
    """Load a PDF document from AWS S3.

    Args:
        filename (str): The filename of the PDF document.

    Returns:
        fitz.Document: The loaded PDF document.
    """
    # Load the PDF from the S3 object
    try:
        data = load_data_from_aws(filename)
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
    data = load_data_from_aws(filename)

    # Convert the PNG data to an image using PIL
    image = Image.open(io.BytesIO(data))

    # Convert the PIL image to a NumPy array
    return np.array(image)


def load_data_from_aws(filename: str) -> bytes:
    """Load a PNG image from AWS S3.

    Args:
        filename (str): The filename of the PNG image.

    Returns:
        bytes: The loaded PNG image.
    """
    # Check if the PNG exists in S3
    try:
        png_object = s3_client.get_object(Bucket=config.bucket_name, Key=filename)
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Document not found in S3 bucket.") from None

    # Load the PNG from the S3 object
    try:
        data = png_object["Body"].read()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load data.") from None

    return data
