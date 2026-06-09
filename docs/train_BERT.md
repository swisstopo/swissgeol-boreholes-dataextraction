# Train BERT Model

To fine-tune BERT on your data, follow these steps:

## 1. Prepare Data
Training requires a JSON file containing layer descriptions and their ground truth classes.

Each file contains boreholes, and each borehole contains layers. For each layer:
- `material_description` - input text for BERT
- a classification label depending on the system:

# TODO: update table
| Class | Config string | JSON layer tag |
|-------|--------------|----------------|
| `ColorSystem` | `color` | - |
| `ColorConsolidatedSystem` | `color_consolidated` | `consolidated.primary_color` |
| `ColorUnconsolidatedSystem` | `color_unconsolidated` | `unconsolidated.primary_color` |
| `DebrisUnconsolidatedSystem` | `debris` | `unconsolidated.debris` |
| `ENMainSystem` | `en_main` | `unconsolidated.main` |
| `LithologySystem` | `lithology` | `consolidated.lithology` |
| `MineralComponentsSystem` | `mineral_components` | `consolidated.mineral_components` |
| `OrganicComponentsUnconsolidatedSystem` | `organic_components` | `unconsolidated.organic_components` |
| `USCSSystem` | `uscs` | `unconsolidated.uscs` |


An example json can be found in [groundtruth-json.md](groundtruth-json.md).

## 2. Choose Hyperparameters

Modify the file `config/bert_config_uscs.yml` to set the hyperparameters for training and data processing. Data sources used for training and evaluation are specified in this file.
It looks like this:

```yml
classification_system: "uscs"
experiment_name: "uscs"
model_path: "google-bert/bert-base-multilingual-uncased"
use_class_balancing: false

# Training and testing sets
training_sets:
  uscs:
    classification_system: uscs
    ground_truths:
      - deepwells_ground_truth.json
      - geoquat_ground_truth.json

test_sets:
  uscs:
    classification_system: uscs
    ground_truths:
      - deepwells_ground_truth.json
      - geoquat_ground_truth.json

# Training hyperparameters
hyperparameters:
  batch_size: 32
  num_epochs: 32
  learning_rate: 1e-4
  weight_decay: 0.001
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine_with_restarts"
  max_grad_norm: 5.0

# Layers to fine-tune
unfreeze_layers:
  - "classifier"
  - "pooler"
  - "layer_11"
```

Each entry under `training_sets` and `test_sets` is a named dataset with:
- `classification_system`: one of "`color_consolidated`, `color_unconsolidated`, `en_main`, `lithology`, `mineral_components`, or `uscs`
- `ground_truths`: list of JSON filenames relative to the project data path

Multiple named datasets can be listed under `training_sets` — their samples are pooled for training. Each entry in `test_sets` is evaluated independently and produces its own metrics report. The train/test split is deterministic per filename, so the same ground-truth files can safely appear in both sections without data leakage.

## 3. Train the Model

To fine-tune BERT from the base model on Hugging Face, run:

```bash
fine-tune-bert -cf bert_config_uscs.yml
```

- Use `-cf` or `--config-file-path` to specify the config file containing the training parameters.
- By default, the initial model is the one specified in `model_path` in the config file (loaded from Hugging Face). To resume training from a checkpoint, pass the checkpoint folder with `-c` or `--model-checkpoint`:

```bash
fine-tune-bert -cf bert_config_uscs.yml -c models/uscs/20240101-120000/checkpoint-500
```

The pipeline saves a checkpoint after each epoch and logs training details in the `models` directory. The run folder name corresponds to the timestamp when training was launched. After training completes, intermediate checkpoints are removed and only the final fine-tuned model head is kept.
