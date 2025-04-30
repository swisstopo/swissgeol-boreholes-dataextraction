import yaml
import csv
import json
from pathlib import Path

def load_csv_to_yaml(csv_path: Path, yaml_path: Path, filter_term_order: int = None):
    """Transform lithology data from CSV to a structured dictionary (yml).

    Keep both the prefered and alternative patterns, and description for each language.

    Args:
        csv_path (str): Path to the CSV file to process
        yaml_path (str): Path to the output YAML file
        term_order (int): Filter by term order (optional)
    """

    languages = ['en', 'fr', 'de', 'it']
    lithology_patterns = {lang: {} for lang in languages}

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if filter_term_order is not None:
                if int(row.get('term_order', 0)) != filter_term_order:
                    continue

            pattern_id = row['id']
            description = row['definition'] if row['definition'] and row['definition'] != 'nan' else None

            for lang in languages:
                # Get the preferred label for this language
                pref_lable = row.get(f'prefLabel_{lang}')

                # Handle alternative labels
                alt_lables = []
                alt_lable_raw = row.get(f'altLabel_{lang}')
                if alt_lable_raw and alt_lable_raw != '':
                     formatted_alt_label = alt_lable_raw.replace("'", '"')
                     alt_lables = json.loads(formatted_alt_label)
                else:
                    alt_lables = []

                lables = [pref_lable] if pref_lable else []
                if alt_lables:
                    lables = lables + alt_lables

                # Store in the patterns dictionary
                lithology_patterns[lang][pattern_id] = {
                    'labels': lables,
                    'description': description
                }

    descriptions_output = yaml.dump(lithology_patterns, allow_unicode=True, sort_keys=False, default_flow_style=None)

    with open(yaml_path, 'w', encoding='utf-8') as file:
        file.write(descriptions_output)

    print(f"YML file saved to: {yaml_path}")

if __name__ == "__main__":
    load_csv_to_yaml()