"""Second script to convert ground truth file into the second new format (with list of boreholes)."""

import json

import click


def transform_json(input_json):
    """Transforms the input JSON structure to ensure each PDF entry contains a list of boreholes.

    Args:
        input_json (dict): The original JSON structure.

    Returns:
        dict: The transformed JSON structure.
    """
    output_json = {}

    for pdf, borehole_data in input_json.items():
        # Wrap the borehole data into a list with an index
        output_json[pdf] = [{"borehole_index": 0, **borehole_data}]

    return output_json


@click.command()
@click.option("-g", "--ground_truth_path", type=click.Path(exists=True))
@click.option("-o", "--out_path", type=click.Path())
def convert_ground_truth_v2(ground_truth_path, out_path):
    """CLI command to transform borehole JSON structure and save the output.

    Probably works only after the first convert_ground_truth.py has been applied.

    Args:
        ground_truth_path: Path to the input JSON file.
        out_path: Path to save the transformed JSON file.

    Usage:
    ```
    python src/scripts/convert_ground_truth_v2.py -g ./data/geoquat_old_ground_truth.json \
    -o ./data/geoquat_new_ground_truth.json
    ```
    """
    with open(ground_truth_path) as file:
        data = json.load(file)

    transformed_data = transform_json(data)

    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(transformed_data, file, indent=4, ensure_ascii=False)

    click.echo(f"Transformation complete. Output saved to {out_path}")


if __name__ == "__main__":
    convert_ground_truth_v2()
