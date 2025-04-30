"""Script that loads the rdf file from the swissgeol-lexic-vocabulary-lithologie repository and creates a csv file."""

import csv
import json
import os
import re
from pathlib import Path

import rdflib
import requests


def extract_id(url: str) -> str:
    """Returns the last part of the URL.

    Args:
        url (str): URL to extract the id from.

    Returns:
        str: returns the last part of the URL
    """
    if isinstance(url, str):
        return url.split("/")[-1]
    return url


def count_capital_terms(id_value: str) -> int:
    """Count the number of capital terms in a given string.

    Args:
        id_value (str): The string to count capital terms in.

    Returns:
        int: The number of capital terms in the string.
    """
    terms = re.findall(r"[A-Z][a-z0-9]*", id_value)
    return len(terms)


def load_rdf_str(rdf_url) -> str:
    """Load an RDF file from a remote URL and return its content as a string.

    Args:
        rdf_url (str): the url of the remote URL.

    Returns:
        str: The content of the RDF file as a string.
    """
    response = requests.get(rdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch RDF file. Status code: {response.status_code}")
    return response.text


def write_rdf(rdf_data, output_dir, rdf_file_name):
    """Write RDF data to a file, creating the output directory if it doesn't exist.

    Args:
        rdf_data (str): The RDF content to write into the file.
        output_dir (str): Path to the directory where the RDF file should be saved.
        rdf_file_name (str): Name of the RDF file to create.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, rdf_file_name), "w", encoding="utf-8") as f_rdf:
        f_rdf.write(rdf_data)


def load_rdf_to_csv(
    output_dir: Path = Path("data/lithology_lexic"),
    output_rdf_name: str = "lithology.rdf",
    output_csv_name: str = "lithology.csv",
):
    """Load RDF file from the swissgeol-lexic-vocabulary-lithologie repository and create a CSV file.

    Args:
        output_dir (Path): folder to store the output files in.
        output_rdf_name (str): name of the output rdf file.
        output_csv_name (str): name of the output csv file.
    """
    # Settings
    SUPPORTED_LANGUAGES = ["en", "de", "fr", "it"]
    rdf_url = "https://raw.githubusercontent.com/swisstopo/swissgeol-lexic-vocabulary-lithologie/main/lithology.rdf"

    # Download RDF
    rdf_data = load_rdf_str(rdf_url)
    write_rdf(rdf_data, output_dir, output_rdf_name)

    # Parse RDF
    graph = rdflib.Graph()
    graph.parse(data=rdf_data, format="xml")

    # Prepare rows
    rows = []

    for s in graph.subjects(rdflib.RDF.type, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#Concept")):
        url = extract_id(str(s))
        row = {
            "id": url,
            "term_order": count_capital_terms(url),
            "prefLabel_en": None,
            "prefLabel_fr": None,
            "prefLabel_de": None,
            "prefLabel_it": None,
            "definition": None,
            "altLabel_en": [],
            "altLabel_fr": [],
            "altLabel_de": [],
            "altLabel_it": [],
            "broader": [],
            "narrower": [],
        }

        for p, o in graph.predicate_objects(subject=s):
            p_str = str(p)

            # Handle prefLabel
            if p_str == "http://www.w3.org/2004/02/skos/core#prefLabel":
                lang = o.language
                if lang in SUPPORTED_LANGUAGES:
                    row[f"prefLabel_{lang}"] = str(o)
                else:
                    raise ValueError(f"Unknow language: {lang}")

            # Handle altLabel
            elif p_str == "http://www.w3.org/2004/02/skos/core#altLabel":
                lang = o.language
                if lang in SUPPORTED_LANGUAGES:
                    row[f"altLabel_{lang}"].append(str(o))
                else:
                    raise ValueError(f"Unknow language: {lang}")

            # Handle definition
            elif p_str == "http://www.w3.org/2004/02/skos/core#definition":
                assert o.language == "en"  # only one definition in english is supported
                row["definition"] = str(o)

            # Handle broader
            elif p_str == "http://www.w3.org/2004/02/skos/core#broader":
                row["broader"].append(extract_id(str(o)))

            # Handle narrower
            elif p_str == "http://www.w3.org/2004/02/skos/core#narrower":
                row["narrower"].append(extract_id(str(o)))

        rows.append(row)

    # Write CSV
    output_csv_path = Path(output_dir, output_csv_name)
    with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Convert dict fields to JSON strings for storage in CSV
            for field in ["broader", "narrower"]:
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)

    print(f"CSV file saved to: {output_csv_path}")


if __name__ == "__main__":
    load_rdf_to_csv()
