"""Get a subset of nagra."""

import json
import random


def create_subset(input_file, output_file, fraction=0.1):
    """Get fraction of dataset."""
    # Read the big JSON file
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object (dictionary) at the root.")

    # Get the keys and pick the first 10% (or random 10%, you can decide)
    keys = list(data.keys())
    subset_size = max(1, int(len(keys) * fraction))
    randomize = False
    selected_keys = random.sample(keys, subset_size) if randomize else keys[:subset_size]

    # Create a subset dictionary
    subset = {k: data[k] for k in selected_keys}

    # Write the subset to a new JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    create_subset("data/nagra_ground_truth.json", "data/nagra_subset.json")
