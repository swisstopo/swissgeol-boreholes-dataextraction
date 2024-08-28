"""Utils related to logging."""

import logging

from app.common.config import config


def setup_logging() -> None:
    """Sets up logging config. Needs to be called at the startup of the app."""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"simple": {"format": "%(asctime)s - %(levelname)s: %(message)s"}},
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"root": {"level": config.logging_level, "handlers": ["stdout"]}},
    }

    logging.config.dictConfig(log_config)  # type: ignore[attr-defined]


def get_app_logger() -> logging.Logger:
    """Utility method to get the app logger."""
    return logging.getLogger("pentagon")
