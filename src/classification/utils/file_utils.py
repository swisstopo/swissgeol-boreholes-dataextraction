"""This module contains general utility functions."""

from importlib import resources
from pathlib import Path

import yaml


def find_project_root() -> Path:
    """Find project root by looking for marker files.

    The base root location is based on the presence of a pyproject.toml **or**
    a setup.py file.

    ```
    project-root
    ├── pyproject.toml
    ├── setup.py
    └── ...
    ```

    Returns:
        Path: Detected root path, or current parent if not detected.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        # Look for typical project root markers
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent
    return current.parents[2]  # Fallback


def read_params(config_filename: str) -> dict:
    """Read parameters from config file.

    First tries config.

    ```
    config
    ├── xxx_params.yml
    ├── yyy_params.yml
    ├── ...
    └── zzz_params.yml
    ```

    Then falls back to package defaults.

    ```
    package
    └── config
        ├── xxx_params.yml
        ├── yyy_params.yml
        ├── ...
        └── zzz_params.yml
    ```

    Args:
        config_filename (str): Name of the config yaml file.
    """
    # Try user config first
    config_path = find_project_root() / "config" / config_filename

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Fall back to package defaults
    try:
        config_data = resources.files("classification").joinpath(f"config/{config_filename}").read_text()
        return yaml.safe_load(config_data)
    except Exception as e:
        raise FileNotFoundError(f"Config {config_filename} not found") from e
