"""Script that loads the rdf file from the swissgeol-lexic-vocabulary-lithologie repository and creates a csv file."""

import csv
import json
import os
import re
from pathlib import Path

import rdflib
import requests


def extract_id(url: str):
    """Returns the last part of the URL.

    Args:
        url (str): URL to extract the id from.

    Returns:
        str: returns the last part of the URL
    """
    if isinstance(url, str):
        return url.split("/")[-1]
    return url


def count_capital_terms(id_value: str):
    """Count the number of capital terms in a given string.

    Args:
        id_value (str): The string to count capital terms in.

    returns:
        int: The number of capital terms in the string.
    """
    terms = re.findall(r"[A-Z][a-z0-9]*", id_value)
    return len(terms)


def process_url_array(url_array_str: str):
    """Process a string representation of a list of URLs.

    Args:
        url_array_str (str): String representation of a list of URLs.

    returns:
        list: List of the last parts of the URLs.
    """
    if not url_array_str or url_array_str == "[]":
        return []

    try:
        # Convert string representation to list
        url_array_str = url_array_str.replace("'", '"')
        url_array = json.loads(url_array_str)

        # Extract the last part of each URL
        return [url.split("/")[-1] for url in url_array]
    except json.JSONDecodeError:
        return []


def load_rdf_to_csv(output_csv: Path):
    """Load RDF file from the swissgeol-lexic-vocabulary-lithologie repository and create a CSV file.

    Args:
        output_csv (Path): Path to the output CSV file.
    """
    # Settings
    rdf_url = "https://raw.githubusercontent.com/swisstopo/swissgeol-lexic-vocabulary-lithologie/main/lithology.rdf"
    SUPPORTED_LANGUAGES = ["en", "de", "fr", "it"]

    # Download RDF/XML
    output_dir = "data/lithology_lexic"
    response = requests.get(rdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch RDF file. Status code: {response.status_code}")
    rdf_data = response.text
    with open(os.path.join(output_dir, "lithology.rdf"), "w", encoding="utf-8") as f_rdf:
        f_rdf.write(rdf_data)

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
    fieldnames = [
        "id",
        "term_order",
        "prefLabel_en",
        "prefLabel_fr",
        "prefLabel_de",
        "prefLabel_it",
        "definition",
        "altLabel_en",
        "altLabel_fr",
        "altLabel_de",
        "altLabel_it",
        "broader",
        "narrower",
    ]

    with open(output_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Convert dict fields to JSON strings for storage in CSV
            for field in ["broader", "narrower"]:
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)

    print(f"CSV file saved to: {output_csv}")


if __name__ == "__main__":
    load_rdf_to_csv()
