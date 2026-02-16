"""Normalize MLflow tracking import."""

import os

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"
mlflow = None

if mlflow_tracking:
    import mlflow  # noqa: F401
