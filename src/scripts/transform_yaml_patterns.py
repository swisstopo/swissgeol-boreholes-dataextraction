"""Script to transform the initial yaml file used for the baseline classifier to a yaml for the Bedrock classifier."""

import yaml

in_file = "config/lithology_classification_params.yml"
out_file = "config/bedrock/lithology_classification_patterns_bedrock.yml"

unconsolidated = ["clay", "marl", "silt", "peat", "sand", "pebble", "loam"]

# Load your input YAML
with open(in_file, encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Prepare the output structure
output = {"baseline": {lang: {} for lang in data["lithology_patterns"]}}
for lang, classes in data["lithology_patterns"].items():
    for rock, info in classes.items():
        if rock.lower() in unconsolidated:
            continue
        labels = info["labels"]
        description = info["description"]

        label_str = labels[0] if len(labels) == 1 else f"{labels[0]} ({', '.join(labels[1:])})"

        # Set the new format
        output["baseline"][lang][rock] = f"{label_str}: {description}"
    output["baseline"][lang]["Unconsolidated"] = (
        f"unconsolidated material: Any unconsolidated material like {', '.join(unconsolidated)}"
    )

# Save the transformed YAML
with open(out_file, "w", encoding="utf-8") as f:
    yaml.dump(output, f, sort_keys=False, allow_unicode=True, width=1000)

print(f"Transformation complete. See {out_file}.")
