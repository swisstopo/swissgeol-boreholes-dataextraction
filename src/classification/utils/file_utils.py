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


def read_classification_params(config_filename: str) -> dict:
    """Read parameters from a yaml file.

    Args:
        config_filename (str): Name of the params yaml file.
    """
    config_path = find_project_root() / "config" / config_filename

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