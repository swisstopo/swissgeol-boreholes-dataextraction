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


config = Config()
