"""Backend configurations."""

import logging
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_aws_bucket_name() -> str:
    """Get the AWS bucket name."""
    return os.getenv("AWS_S3_BUCKET") if os.getenv("AWS_S3_BUCKET") else "stijnvermeeren-boreholes-integration-tmp"


def get_aws_endpoint() -> str:
    """Get the AWS endpoint."""
    bucket_name = get_aws_bucket_name()
    endpoint_name = os.getenv("AWS_ENDPOINT")

    if not endpoint_name:
        raise ValueError("AWS_ENDPOINT environment variable is not set.")

    endpoint_str = f"https://{bucket_name}.{endpoint_name.replace('https://', '')}" if endpoint_name else None

    return endpoint_str


class Config(BaseSettings):
    """Configuration for the backend."""

    model_config = SettingsConfigDict(env_prefix="BOREHOLE_")

    ###########################################################
    # Logging
    ###########################################################
    logging_level: int = logging.DEBUG

    ###########################################################
    # AWS Settings
    ###########################################################
    bucket_name: str = get_aws_bucket_name()
    test_bucket_name: str = "test-bucket"

    ###########################################################
    # AWS Credentials
    ###########################################################
    aws_access_key_id: str | None = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_endpoint: str | None = get_aws_endpoint()


config = Config()
