"""Backend configurations."""

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration for the backend."""

    model_config = SettingsConfigDict(env_prefix="BOREHOLE_")

    ###########################################################
    # Logging
    ###########################################################
    logging_level: int = logging.DEBUG

    ###########################################################
    # AWS
    ###########################################################
    bucket_name: str = "stijnvermeeren-boreholes-data"

    # TODO: check how this is used on the VM
    # aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    # aws_secret_key_access = os.environ.get("AWS_SECRET_ACCESS_KEY")
    # aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    # aws_endpoint = os.environ.get("AWS_ENDPOINT")


config = Config()
