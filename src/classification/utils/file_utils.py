"""File utilities for classification project."""

from importlib import resources
from pathlib import Path

import yaml


def find_project_root() -> Path:
    """Find project root by looking for marker files like pyproject.toml or setup.py.

    Returns:
        Path: The project root directory.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        # Look for typical project root markers
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent


def read_classification_params(config_filename: str, user_config_path: Path = None) -> dict:
    """Read parameters from config file.

    First tries user_config_path, then falls back to package defaults.

    Args:
        config_filename (str): Name of the config yaml file.
        user_config_path (Path, optional): Path to user-provided config file. Defaults to None.
    """
    # Try user-provided config first
    if user_config_path:
        config_path = find_project_root() / user_config_path / config_filename

        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

    # Fall back to package defaults
    try:
        config_data = resources.files("swissgeol_doc_processing").joinpath(f"config/{config_filename}").read_text()
        return yaml.safe_load(config_data)
    except Exception as e:
        raise FileNotFoundError(f"Config {config_filename} not found") from e
