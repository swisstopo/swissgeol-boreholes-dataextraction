"""This module contains general utility functions."""

import functools
import re
import shutil
import time
from collections.abc import MutableMapping
from importlib import resources
from pathlib import Path

import yaml


def expose_configs() -> None:
    """Copy the package's default configuration folder into the project's root-level `config` directory.

    This function is intended for users who want to customize configuration files locally.
    It copies the package configuration:

    ```
    package
    └── config
        ├── xxx_params.yml
        ├── yyy_params.yml
        ├── ...
        └── zzz_params.yml
    ```

    To the project root folder.

    ```
    project-root
    └── config
        ├── xxx_params.yml
        ├── yyy_params.yml
        ├── ...
        └── zzz_params.yml
    ```

    Args:
        package (str): Package name that contains configuration parameters

    Raises:
        FileNotFoundError: If the package's config folder cannot be located.
        OSError: If copying fails for system-related reasons.
    """
    # Find project root
    project_root = find_project_root()
    destination_folder = project_root / "config"
    destination_folder.mkdir(parents=True, exist_ok=True)

    # Resolve the package config folder
    source_folder = resources.files("swissgeol_doc_processing").joinpath("config")

    # Perform copy
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)


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


def read_params(
    config_filename: str,
) -> dict:
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
    # Try user-provided config first
    config_path = find_project_root() / "config" / config_filename

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Fall back to package defaults
    try:
        config_data = resources.files("swissgeol_doc_processing").joinpath(f"config/{config_filename}").read_text()
        return yaml.safe_load(config_data)
    except Exception as e:
        raise FileNotFoundError(f"Config {config_filename} not found") from e


def flatten(dictionary: dict, parent_key: str = "", separator: str = "__") -> dict:
    """Flatten a nested dictionary.

    Args:
        dictionary (Dict): Dictionary to flatten.
        parent_key (str, optional): Prefix for flattened key. Defaults to ''.
        separator (str, optional): The separator used when concatenating keys. Defaults to '__'.

    Returns:
        Dict: Flattened dictionary.
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def parse_text(text: str) -> str:
    """Parse text by removing non-alphanumeric characters and converting to lowercase.

    Args:
        text (str): Text to parse.

    Returns:
        str: Parsed text.
    """
    not_alphanum = re.compile(r"[^\w\d]", re.U)
    return not_alphanum.sub("", text).lower() if text else ""


def timeit(func):
    """Compute running time of function."""

    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper
