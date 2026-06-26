"""Normalize W&B tracking import."""

import os

wandb_tracking = os.getenv("WANDB_TRACKING") == "True"
wandb = None

if wandb_tracking:
    import wandb  # noqa: F401
