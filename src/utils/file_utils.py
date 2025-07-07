"""This module contains general utility functions."""

import functools
import re
import time
from collections.abc import MutableMapping

import yaml
from pyinstrument import Profiler

from extraction import PROJECT_ROOT


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


def read_params(config_filename: str) -> dict:
    """Read parameters from a yaml file.

    Args:
        config_filename (str): Name of the params yaml file.
    """
    config_path = PROJECT_ROOT / "config" / config_filename

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


def profile(func):
    """Decorator that profiles a function using pyinstrument and prints the report.

    Usage:
        @profile
        def your_function():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        return result

    return wrapper
