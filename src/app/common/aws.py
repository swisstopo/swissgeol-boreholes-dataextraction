"""Series of utility functions for the aws connection to get the groundwater stratigraphy."""

import io
import os
from pathlib import Path

import boto3
import numpy as np
import pymupdf
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
from dotenv import load_dotenv
from fastapi import HTTPException
from PIL import Image

from app.common.config import DEFAULT_BUCKET_NAME, config
from app.common.log import get_app_logger

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
    _test_s3_client(_s3_client)
    return _s3_client


def create_s3_client():
    """Create an S3 client using default or custom credentials.

    Returns:
        boto3.client: The S3 client.
    """
    if not config.aws_endpoint:
        logger.warning(
            "No endpoint provided, a client with an empty endpoint value might have an unexpected behaviour."
        )
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            endpoint_url=config.aws_endpoint,
        )
        return s3_client
    except (NoCredentialsError, ClientError) as e:
        logger.error(f"Error accessing S3 with custom credentials: {e}")
        raise HTTPException(status_code=500, detail="Failed to access S3.") from None


def _test_s3_client(s3_client: boto3.client):
    """Test the s3 client by trying a simple operation.

    Args:
        s3_client (boto3.client): The S3 client.
    """
    if config.bucket_name is None:
        raise HTTPException(status_code=500, detail="AWS S3 bucket name must be defined.") from None

    try:
        # Test bucket access with a lightweight operation
        s3_client.head_bucket(Bucket=config.bucket_name)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        http_status = e.response["ResponseMetadata"]["HTTPStatusCode"]

        logger.error(f"S3 ClientError - Code: {error_code}, HTTP: {http_status}, Message: {error_message}")

        if http_status == 403 and config.bucket_name:
            raise HTTPException(
                status_code=403, detail=f"Access denied to the bucket {config.bucket_name}. Check the AWS permissions."
            ) from None
        elif http_status == 404 and config.bucket_name:
            raise HTTPException(status_code=404, detail=f"Bucket {config.bucket_name} does not exist.") from None
        else:
            raise HTTPException(
                status_code=500, detail=f"AWS Client error: {error_code}, message: {error_message}"
            ) from None

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
    if not str(filename).lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid request. The filename must end with '.pdf'.")
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
        if not os.getenv("AWS_S3_BUCKET"):
            # logging is here to avoid circular imports error in the config module
            logger.warning(f"No bucket name provided, defaulting to {DEFAULT_BUCKET_NAME}")
        s3_object = s3_client.get_object(Bucket=config.bucket_name, Key=str(prefix / filename))
    except s3_client.exceptions.NoSuchKey:
        # Special case for PNG files - they need to be generated first
        if str(filename).lower().endswith(".png") and prefix == "dataextraction":
            raise HTTPException(
                status_code=404,
                detail=(
                    f"PNG file {prefix / filename} not found. "
                    "The PNG files need to be generated first using the create_pngs endpoint."
                ),
            ) from None
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


# Files required for BERT inference. Training artifacts (optimizer.pt, scheduler.pt, etc.) are excluded.
_BERT_INFERENCE_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]


def _get_model_bucket() -> str:
    """Return the S3 bucket name for BERT models.

    Uses BERT_MODEL_S3_BUCKET if set, otherwise falls back to the main data bucket.
    """
    return os.environ.get("BERT_MODEL_S3_BUCKET") or config.bucket_name


def download_model_from_s3(s3_key_prefix: str, local_dir: Path) -> Path:
    """Download BERT model inference files from S3 to a local directory.

    Already-present files are skipped, so warm container restarts reuse previously
    downloaded weights without hitting S3 again.

    Args:
        s3_key_prefix (str): S3 folder prefix within the model bucket,
            e.g. "lithology_models/best_model_lithology".
        local_dir (Path): Local directory to download files into (created if absent).

    Returns:
        Path: local_dir, ready to pass to AutoModelForSequenceClassification.from_pretrained.

    Raises:
        HTTPException: If a required file is missing in S3 or the download fails.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    s3_client = get_s3_client()
    bucket = _get_model_bucket()

    for filename in _BERT_INFERENCE_FILES:
        local_path = local_dir / filename
        if local_path.exists():
            continue
        s3_key = f"{s3_key_prefix}/{filename}"
        try:
            s3_client.download_file(bucket, s3_key, str(local_path))
            logger.info(f"Downloaded s3://{bucket}/{s3_key} → {local_path}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchKey"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Model file '{s3_key}' not found in bucket '{bucket}'.",
                ) from None
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download model file '{s3_key}': {e}",
            ) from None

    return local_dir
