"""Normalize W&B tracking import."""

import os

from dotenv import load_dotenv

load_dotenv()

wandb_tracking = os.getenv("WANDB_TRACKING") == "True"
wandb = None

if wandb_tracking:
    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError as e:
        raise ImportError("wandb is not installed. Run: uv sync --extra wandb") from e
