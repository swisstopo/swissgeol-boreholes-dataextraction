"""Series of utility functions for the aws connection to get the groundwater stratigraphy."""

import io
from pathlib import Path

import boto3
import numpy as np
import pymupdf
from app.common.config import config
from app.common.log import get_app_logger
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
from dotenv import load_dotenv
from fastapi import HTTPException
from PIL import Image

load_dotenv()
logger = get_app_logger()

_s3_client = None  # Global reference to the S3 client


def get_s3_client():
    """Lazy initialization of the S3 client.

    Returns:
        boto3.client: The S3 client.
    """
    global _s3_client
    if _s3_client is None:
        _s3_client = create_s3_client()
    _s3_test_client(_s3_client)
    return _s3_client


def create_s3_client():
    """Create an S3 client using default or custom credentials.

    Returns:
        boto3.client: The S3 client.
    """
    if not config.aws_endpoint:
        print("No endpoint provided, a client with an empty endpoint value might have an unexpected behaviour.")
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            endpoint_url=config.aws_endpoint,
        )
        return s3_client
    except (NoCredentialsError, ClientError) as e:
        print(f"Error accessing S3 with custom credentials: {e}")
        raise HTTPException(status_code=500, detail="Failed to access S3.") from None


def _s3_test_client(s3_client: boto3.client):
    """Test the s3 client by trying a simple operation.

    Args:
        s3_client (boto3.client): The S3 client.
    """
    try:
        s3_client.list_buckets()
        return None

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ["InvalidAccessKeyId", "SignatureDoesNotMatch"]:
            raise HTTPException(status_code=401, detail="Invalid AWS credentials.") from None
        else:
            raise HTTPException(status_code=500, detail=f"AWS Client error: {error_code}") from None
    except EndpointConnectionError as e:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: could not connect to S3 endpoint URL: "
            f"{e.kwargs.get('endpoint_url', 'Unknown')}",
        ) from None


def load_pdf_from_aws(filename: Path) -> pymupdf.Document:
    """Load a PDF document from AWS S3.

    Args:
        filename (str): The filename of the PDF document.

    Returns:
        pymupdf.Document: The loaded PDF document.
    """
    # Load the PDF from the S3 object
    data = load_data_from_aws(filename)
    return pymupdf.open(stream=data, filetype="pdf")


def load_png_from_aws(filename: Path) -> np.ndarray:
    """Load a PNG image from AWS S3.

    Args:
        filename (str): The filename of the PNG image.

    Returns:
        ndarray: The loaded PNG image.
    """
    data = load_data_from_aws(filename, "dataextraction")

    # Convert the PNG data to an image using PIL
    image = Image.open(io.BytesIO(data))

    # Convert the PIL image to a NumPy array
    return np.array(image)


def load_data_from_aws(filename: Path, prefix: str = "") -> bytes:
    """Load a document from AWS S3.

    Args:
        filename (str): The filename of the document.
        prefix (str): The prefix of the file in the bucket.

    Returns:
        bytes: The loaded document.
    """
    s3_client = get_s3_client()

    # Check if the document exists in S3
    try:
        s3_object = s3_client.get_object(Bucket=config.bucket_name, Key=str(prefix / filename))
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Document {prefix / filename} not found in S3 bucket.") from None

    # Load the document from the S3 object
    try:
        data = s3_object["Body"].read()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load data.") from None

    return data


def upload_file_to_s3(file_path: str, key: str):
    """Upload a file to S3.

    Args:
        file_path (str): The local path to the file to upload.
        key (str): The key (name) of the file in the bucket.
    """
    get_s3_client().upload_file(file_path, config.bucket_name, key)
