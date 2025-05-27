"""Backend configurations."""

import logging
import os

import dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv(override=True)


def get_aws_bucket_name() -> str:
    """Get the AWS bucket name."""
    bucket_name = os.getenv("AWS_S3_BUCKET")
    default_name = "stijnvermeeren-boreholes-integration-tmp"
    if not bucket_name:
        print(f"No bucket name provided, defaulting to {default_name}")
        return default_name
    return bucket_name


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
    aws_endpoint: str | None = os.getenv("AWS_ENDPOINT")


config = Config()
