"""Script to create train-test-val split with the layers contained in a list of ground truth files."""

import json
import os
import random


def load_layers(json_paths: list[str]) -> list[tuple[str, int, dict, dict]]:
    """Load layers along with their associated file name, borehole index, and metadata from multiple JSON files.

    Args:
        json_paths (list[str]): List of paths to JSON files.

    Returns:
        list[tuple[str, int, dict, dict]]: List of tuples (file name, borehole index, layer dictionary, metadata).
    """
    all_layers = []
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            for file_name, boreholes in data.items():
                for borehole in boreholes:
                    borehole_index = borehole.get("borehole_index", 0)
                    metadata = borehole.get("metadata", {})
                    for layer in borehole.get("layers", []):
                        all_layers.append((file_name, borehole_index, layer, metadata))
    return all_layers


def split_layers(
    layers: list[tuple[str, int, dict, dict]], train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42
) -> tuple[list[tuple[str, int, dict, dict]], list[tuple[str, int, dict, dict]], list[tuple[str, int, dict, dict]]]:
    """Split a list of layers into train, validation, and test subsets.

    Args:
        layers (list[tuple[str, int, dict, dict]]): List of (file name, borehole index, layer, metadata).
        train_frac (float): Fraction of layers to assign to the training set.
        val_frac (float): Fraction of layers to assign to the validation set. The test fraction is implicitly
            `1 - train_frac - val_frac`.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[list[tuple[]]: Three lists `(train_layers, val_layers, test_layers)` containing the layer records.
    """
    random.seed(seed)
    shuffled_layers = layers.copy()
    random.shuffle(shuffled_layers)

    n_total = len(shuffled_layers)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_layers = shuffled_layers[:n_train]
    val_layers = shuffled_layers[n_train : n_train + n_val]
    test_layers = shuffled_layers[n_train + n_val :]

    return train_layers, val_layers, test_layers


def reconstruct_structure(layers: list[tuple[str, int, dict, dict]]) -> dict[str, list[dict]]:
    """Reconstruct the original JSON structure from layers with filenames, borehole indices, and metadata.

    Args:
        layers (list[tuple[str, int, dict, dict]]): List of (file name, borehole index, layer, metadata).

    Returns:
        dict[str, list[dict]]: Dictionary ready to be saved in the original JSON structure format.
    """
    reconstructed = {}

    for file_name, borehole_index, layer, metadata in layers:
        if file_name not in reconstructed:
            reconstructed[file_name] = []

        # Find existing borehole by index or create it
        boreholes = reconstructed[file_name]
        borehole = next((b for b in boreholes if b["borehole_index"] == borehole_index), None)
        if borehole is None:
            borehole = {"borehole_index": borehole_index, "groundwater": None, "layers": [], "metadata": metadata}
            boreholes.append(borehole)

        borehole["layers"].append(layer)

    return reconstructed


def save_split(split_data: dict[str, list[dict]], output_path: str) -> None:
    """Save a reconstructed split to a JSON file.

    Args:
        split_data (dict[str, list[dict]]): Structured data to save.
        output_path (str): Path where the JSON file will be written.

    Returns:
        None
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main execution function to load, split, reconstruct, and save the datasets.

    Args:
        None

    Returns:
        None
    """
    json_paths = ["data/deepwells_ground_truth.json", "data/nagra_ground_truth.json"]
    output_dir = "data/nagra_deepwells_splits"

    layers = load_layers(json_paths)
    train, val, test = split_layers(layers, train_frac=0.7, val_frac=0.15, seed=42)

    os.makedirs(output_dir, exist_ok=True)

    train_structure = reconstruct_structure(train)
    val_structure = reconstruct_structure(val)
    test_structure = reconstruct_structure(test)

    save_split(train_structure, os.path.join(output_dir, "train.json"))
    save_split(val_structure, os.path.join(output_dir, "val.json"))
    save_split(test_structure, os.path.join(output_dir, "test.json"))


if __name__ == "__main__":
    main()
