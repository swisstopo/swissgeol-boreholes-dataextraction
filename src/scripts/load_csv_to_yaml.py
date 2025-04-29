import yaml
import csv
from pathlib import Path

def load_csv_to_yaml(csv_path: Path, yaml_path: Path):
    """Transform lithology data from CSV to a structured dictionary (yml).

    Keep both the prefered and alternative patterns, and description for each language.

    Args:
        csv_path (str): Path to the CSV file to process
        yaml_path (str): Path to the output YAML file
    """

    languages = ['en', 'fr', 'de', 'it']
    lithology_patterns = {lang: {} for lang in languages}

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            pattern_id = row['id']
            description = row['definition'] if row['definition'] and row['definition'] != 'nan' else None

            for lang in languages:
                # Get the preferred label for this language
                pref_label = row.get(f'prefLabel_{lang}', '')

                # Handle alternative labels
                alt_label = row.get(f'altLabel_{lang}', '')

                # Build the labels list
                labels = [pref_label] if pref_label else []
                if alt_label and alt_label != '':
                    labels.append(alt_label)

                # Store in the patterns dictionary
                lithology_patterns[lang][pattern_id] = {
                    'labels': labels,
                    'description': description
                }

    descriptions_output = yaml.dump(lithology_patterns, allow_unicode=True, sort_keys=False, default_flow_style=None)

    with open(yaml_path, 'w', encoding='utf-8') as file:
        file.write(descriptions_output)

    print(f"YML file saved to: {yaml_path}")

if __name__ == "__main__":
    load_csv_to_yaml()