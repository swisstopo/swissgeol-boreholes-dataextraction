"""Utility functions for benchmarking."""

from __future__ import annotations

import logging

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the application."""
    logging.basicConfig(format=DEFAULT_FORMAT, level=level, datefmt=DEFAULT_DATEFMT)


def _short_metric_key(k: str) -> str:
    """Drop the first namespace segment.

    Examples:
      geology/layer_f1 -> layer_f1
      metadata/name_f1 -> name_f1
      layer_f1 -> layer_f1

    Args:
        k (str): The original metric key.

    Returns:
        str: The shortened metric key.
    """
    return k.split("/", 1)[1] if "/" in k else k


def _shorten_metric_dict(metrics: dict[str, float]) -> dict[str, float]:
    """Convert flattened metrics to 'short' keys, avoiding collisions.

    Args:
        metrics (dict[str, float]): The original metric dictionary.

    Returns:
        dict[str, float]: The shortened metric dictionary.
    """
    out: dict[str, float] = {}
    for k, v in metrics.items():
        short = _short_metric_key(k)
        if short in out and k != short:
            out[k] = v
        else:
            out[short] = v
    return out
