"""Script to generate a yml file with the labels and descriptions of lithology from a csv file."""

import csv
import json
from pathlib import Path

import click
import yaml
from scripts.load_rdf_to_csv import load_rdf_to_csv


@click.command()
@click.option("--from-rdf", is_flag=True, help="Download from rdf file directly.")
def load_csv_to_yaml(
    output_csv_dir: Path = Path("data/lithology_lexic"),
    output_csv_name: str = "lithology.csv",
    output_yaml_name: str = "lithology_classification_params.yml",
    filter_term_order: int = 1,
    from_rdf: bool = False,
):
    """Transform lithology data from CSV to a structured dictionary (yml).

    Keep both the prefered and alternative patterns, and description for each language.

    Args:
        output_csv_dir (Path): folder to store the files into. It must contain the csv if from_rdf is False.
        output_csv_name (str): Path to the CSV file to process
        output_yaml_name (str): Path to the output YAML file
        filter_term_order (int): Filter by term order (optional)
        from_rdf (bool): to create the yml file and all the files directly, without initially having the csv
    """
    if from_rdf:
        load_rdf_to_csv(output_dir=output_csv_dir, output_csv_name=output_csv_name)

    languages = ["en", "fr", "de", "it"]
    lithology_patterns = {lang: {} for lang in languages}

    with open(Path(output_csv_dir, output_csv_name), encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if filter_term_order is not None and int(row.get("term_order", 0)) != filter_term_order:
                continue

            pattern_id = row["id"]
            description = row["definition"] if row["definition"] and row["definition"] != "nan" else None

            for lang in languages:
                # Get the preferred label for this language
                pref_label = [row.get(f"prefLabel_{lang}")] if row.get(f"prefLabel_{lang}") else []

                # Handle alternative labels
                alt_labels = json.loads(row.get(f"altLabel_{lang}", "[]").replace("'", '"'))

                # Store in the patterns dictionary
                lithology_patterns[lang][pattern_id] = {"labels": pref_label + alt_labels, "description": description}

    # add kA (keine Angabe),  which is not present in the Data
    kA_description = "not specified"
    ka_labels = {
        "de": ["keine Angabe", "unbekannt", "nicht beschrieben"],
        "fr": ["sans indication"],
        "en": ["not specified"],
        "it": ["senza indicazioni"],
    }
    for lang, labels in ka_labels.items():
        lithology_patterns[lang]["kA"] = {"labels": labels, "description": kA_description}

    descriptions_output = yaml.dump(
        {"lithology_patterns": lithology_patterns}, allow_unicode=True, sort_keys=False, default_flow_style=None
    )
    yaml_path = Path("config", output_yaml_name)
    with open(yaml_path, "w", encoding="utf-8") as file:
        file.write(descriptions_output)

    print(f"YML file saved to: {yaml_path}")


if __name__ == "__main__":
    load_csv_to_yaml()
