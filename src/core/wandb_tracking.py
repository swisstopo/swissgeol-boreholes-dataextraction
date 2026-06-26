"""Normalize W&B tracking import."""

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

wandb_tracking = os.getenv("WANDB_TRACKING") == "True"
wandb = None

if wandb_tracking:
    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        import wandb  # noqa: F401
