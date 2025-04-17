"""This module contains general utility functions."""

import re
from collections.abc import MutableMapping

import yaml
from borehole_extraction import PROJECT_ROOT


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


def read_params(params_name: str) -> dict:
    """Read parameters from a yaml file.

    Args:
        params_name (str): Name of the params yaml file.
    """
    config_path = PROJECT_ROOT / "config" / params_name

    if not config_path.exists():
        raise FileNotFoundError(f"Provided parameter file not found: {config_path}")

    if not config_path.is_file():
        raise ValueError(f"Provided path does not point to a file: {config_path}")

    try:
        with open(config_path) as f:
            params = yaml.safe_load(f)

        return params
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML format in {config_path}: {str(e)}") from e


def parse_text(text: str) -> str:
    """Parse text by removing non-alphanumeric characters and converting to lowercase.

    Args:
        text (str): Text to parse.

    Returns:
        str: Parsed text.
    """
    not_alphanum = re.compile(r"[^\w\d]", re.U)
    return not_alphanum.sub("", text).lower()
