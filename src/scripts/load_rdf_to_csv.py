"""Script that loads the rdf file from the swissgeol-lexic-vocabulary-lithologie repository and creates a csv file."""

import csv
import json
import os

import rdflib
import requests

SUPPORTED_LANGUAGES = ["en", "de", "fr", "it"]

# Settings
rdf_url = "https://raw.githubusercontent.com/swisstopo/swissgeol-lexic-vocabulary-lithologie/main/lithology.rdf"
output_dir = "data/lithology_lexic"
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "lithology.csv")

# Download RDF/XML
response = requests.get(rdf_url)
if response.status_code != 200:
    raise Exception(f"Failed to fetch RDF file. Status code: {response.status_code}")
rdf_data = response.text
with open(os.path.join(output_dir, "lithology.rdf"), "w", encoding="utf-8") as f_rdf:
    f_rdf.write(rdf_data)

# Parse RDF
graph = rdflib.Graph()
graph.parse(data=rdf_data, format="xml")

potential_primary = []

# Prepare rows
rows = []

for s in graph.subjects(rdflib.RDF.type, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#Concept")):
    row = {
        "id": str(s),
        "prefLabel_en": None,
        "prefLabel_fr": None,
        "prefLabel_de": None,
        "prefLabel_it": None,
        "definition": None,
        "altLabel": {},
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
            if lang not in row["altLabel"]:
                row["altLabel"][lang] = []
            row["altLabel"][lang].append(str(o))

        # Handle definition
        elif p_str == "http://www.w3.org/2004/02/skos/core#definition":
            assert o.language == "en"  # only one definition in english is supported
            row["definition"] = str(o)

        # Handle broader
        elif p_str == "http://www.w3.org/2004/02/skos/core#broader":
            row["broader"].append(str(o))

        # Handle narrower
        elif p_str == "http://www.w3.org/2004/02/skos/core#narrower":
            row["narrower"].append(str(o))

    rows.append(row)

    # infer primary classes
    short_id = str(s).split("/")[-1]
    if sum(1 for c in short_id if c.isupper()) == 1:
        narrow = [
            str(o).split("/")[-1]
            for p, o in graph.predicate_objects(subject=s)
            if str(p) == "http://www.w3.org/2004/02/skos/core#narrower"
        ]
        if all([sum(1 for c in o if c.isupper()) > 1 for o in narrow]):
            print(short_id)
            potential_primary.append(s)

    #     else:
    #         print(f"     {short_id} is broader than primary (childs={narrow})")
    # elif sum(1 for c in short_id if c.isupper()) == 0:
    #     print("0:", short_id)

    # name = next(
    #     str(o)
    #     for p, o in graph.predicate_objects(subject=s)
    #     if str(p) == "http://www.w3.org/2004/02/skos/core#prefLabel" and o.language == "en"
    # )
    # if ":" not in name:
    #     print(f": not in {name}")

print(f"total potential primary: {len(potential_primary)}")

# filtered_primary = []
# for s in potential_primary:
#     broad = [
#         str(o).split("/")[-1]
#         for p, o in graph.predicate_objects(subject=s)
#         if str(p) == "http://www.w3.org/2004/02/skos/core#broader"
#     ]
#     if any([b in potential_primary for b in broad]):
#         print(f"removed {str(s)}")
#         continue
#     filtered_primary.append(s)


# print(f"total potential primary after filter: {len(filtered_primary)}")

# Write CSV
fieldnames = [
    "id",
    "prefLabel_en",
    "prefLabel_fr",
    "prefLabel_de",
    "prefLabel_it",
    "definition",
    "altLabel",
    "broader",
    "narrower",
]

with open(output_csv, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        # Convert dict fields to JSON strings for storage in CSV
        for field in ["altLabel", "broader", "narrower"]:
            row[field] = json.dumps(row[field], ensure_ascii=False)
        writer.writerow(row)

print(f"CSV file saved to: {output_csv}")
