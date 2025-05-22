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


def load_borehole_reports(json_paths: list[str]) -> list[tuple[str, str, list[dict]]]:
    """Load the reports containing all the layers from multiple JSON files.

    Args:
        json_paths (list[str]): List of paths to JSON files.

    Returns:
        list[tuple[str, str, list[dict]]]: List of tuples (report name, json name, borehole list).
    """
    all_reports = []
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        json_name = path.split("/")[-1].split("_")[0]
        print(json_name)
        for report_name, boreholes in data.items():
            all_reports.append((report_name, json_name, boreholes))
    return all_reports


def split_layers(
    layers: list[tuple[str, int, dict, dict]], train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42
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


def split_reports(reports, train_frac=0.8, val_frac=0.1, eval_sets_nagra_ratio=0.3, seed=42):
    """Split a list of reports into train, validation, and test subsets.

    Note: as we split by reports and not by layers, the required proportion of the splits might not be exact.

    Args:
        reports (list[tuple[str, str, list[dict]]]): List of tuples (report name, json name, report dictionary).
        train_frac (float): Fraction of layers to assign to the training set.
        val_frac (float): Fraction of layers to assign to the validation set. The test fraction is implicitly
            `1 - train_frac - val_frac`.
        eval_sets_nagra_ratio (float): Target maximum proportion of Nagra data in the validation set.
            This constraint exists because the Nagra dataset contains highly similar geological layers.
            Including too many Nagra entries in the evaluation set can lead to unrealistically high metrics
            (e.g., F1 scores close to 1.0), which do not reflect real-world performance on more diverse borehole data.
            The imbalance arises because the Nagra dataset is significantly larger than the Deepwells dataset (~20k
            layers vs. ~4k), which biases the trained model toward recognizing Nagra-style layers more effectively.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[list[tuple[]]: Three lists `(train_layers, val_layers, test_layers)` containing the layer records.
    """
    random.seed(seed)
    shuffled_reports = reports.copy()
    random.shuffle(shuffled_reports)

    def build_set(target_nagra_ratio, min_size):
        tot_layers = 0
        nagra_layers = 0
        data_set = []
        while tot_layers < min_size:
            current_nagra_ratio = nagra_layers / tot_layers if tot_layers != 0.0 else 0.0
            pick_from = "nagra" if current_nagra_ratio < target_nagra_ratio else "deepwells"

            index = shuffled_reports.index(next(r for r in shuffled_reports if r[1] == pick_from))
            rep = shuffled_reports.pop(index)
            rep_num_layers = sum([len(b["layers"]) for b in rep[2]])

            data_set.append(rep)
            tot_layers += rep_num_layers
            nagra_layers += rep_num_layers if pick_from == "nagra" else 0
        print("nagra ratio:", current_nagra_ratio, "number of layers:", tot_layers)
        return data_set

    n_layers = sum([len(b["layers"]) for r in reports for b in r[2]])
    n_train = int(train_frac * n_layers)
    n_val = int(val_frac * n_layers)
    print("validatio set:")
    val_reports = build_set(target_nagra_ratio=eval_sets_nagra_ratio, min_size=n_val)
    print("test set:")
    test_reports = build_set(target_nagra_ratio=eval_sets_nagra_ratio, min_size=n_layers - n_val - n_train)
    train_reports = shuffled_reports.copy()

    return train_reports, val_reports, test_reports


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


def reconstruct_structure_reports(reports):
    """Reconstruct the original JSON structure from layers with filenames, borehole indices, and metadata.

    Args:
        reports (list[tuple[str, str, list[dict]]]): List of tuples (report name, json name, report dictionary).

    Returns:
        dict[str, list[dict]]: Dictionary ready to be saved in the original JSON structure format.
    """
    return {report_name: boreholes for report_name, _, boreholes in reports}


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
    output_dir = "data/lithology_splits"

    reports = load_borehole_reports(json_paths)
    n = len([r for r in reports if r[1] == "nagra"])
    print("total number of reports", len(reports), f"of which {n} are nagra")

    train_reports, val_reports, test_reports = split_reports(
        reports, train_frac=0.8, val_frac=0.1, eval_sets_nagra_ratio=0.3, seed=42
    )

    os.makedirs(output_dir, exist_ok=True)

    train_structure = reconstruct_structure_reports(train_reports)
    val_structure = reconstruct_structure_reports(val_reports)
    test_structure = reconstruct_structure_reports(test_reports)

    save_split(train_structure, os.path.join(output_dir, "train.json"))
    save_split(val_structure, os.path.join(output_dir, "val.json"))
    save_split(test_structure, os.path.join(output_dir, "test.json"))


if __name__ == "__main__":
    main()
