"""Module to load and convert to jsonld the rdf file in swissgeol-lexic-vocabulary-lithologie."""

import json
import os

import rdflib
import requests

# 1. URL of the RDF/XML file
rdf_url = "https://raw.githubusercontent.com/swisstopo/swissgeol-lexic-vocabulary-lithologie/main/lithology.rdf"

# 2. Create directory to store the output
output_dir = "data/lithology_lexic"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "lithology.jsonld")

# 3. Download RDF/XML content
response = requests.get(rdf_url)
if response.status_code != 200:
    raise Exception(f"Failed to fetch RDF file. Status code: {response.status_code}")
rdf_data = response.text

# 4. Parse RDF/XML into an RDF graph
graph = rdflib.Graph()
graph.parse(data=rdf_data, format="xml")

# 5. Serialize the graph to JSON-LD
json_ld_data = graph.serialize(format="json-ld", indent=4)

# 6. Convert to Python dict (optional, for better control)
json_ld_dict = json.loads(json_ld_data)

# 7. Save the result to a JSON-LD file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_ld_dict, f, ensure_ascii=False, indent=4)
