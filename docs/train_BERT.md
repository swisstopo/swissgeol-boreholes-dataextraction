# Train BERT Model

To fine-tune BERT on your data, follow these steps:

## 1. Prepare Data
Training requires a JSON file containing layer descriptions and their ground truth classes.

Each file contains boreholes, and each borehole contains layers. For each layer:
- `material_description` - input text for BERT
- a classification label depending on the system:

| Class | Config string | JSON layer tag |
|-------|--------------|----------------|
| `AccessoryComponentsSystem` | `accessory_components` | `consolidated.accessory_components` |
| `AlterationDegreeSystem` | `alteration_degree` | - |
| `AlterationDegreeConsolidatedSystem` | `alteration_degree_consolidated` | `consolidated.alteration_degree` |
| `AlterationDegreeUnconsolidatedSystem` | `alteration_degree_unconsolidated` | `unconsolidated.alteration_degree` |
| `CementationSystem` | `cementation` | `consolidated.cementation` |
| `ColorSystem` | `color` | - |
| `ColorConsolidatedSystem` | `color_consolidated` | `consolidated.primary_color` |
| `ColorUnconsolidatedSystem` | `color_unconsolidated` | `unconsolidated.primary_color` |
| `DebrisSystem` | `debris` | `unconsolidated.debris` |
| `ENMainSystem` | `en_main` | `unconsolidated.main` |
| `GrainAngularitySystem` | `grain_angularity` | `unconsolidated.grain_angularity` |
| `GrainShapeSystem` | `grain_shape` | `unconsolidated.grain_shape` |
| `LithologySystem` | `lithology` | `consolidated.lithology` |
| `MineralComponentsSystem` | `mineral_components` | `consolidated.mineral_components` |
| `OrganicComponentsSystem` | `organic_components` | `unconsolidated.organic_components` |
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
  learning_rate: 1e-4
  lr_scheduler_type: "cosine_with_restarts"
  max_grad_norm: 5.0
  num_epochs: 32
  warmup_ratio: 0.1
  weight_decay: 0.001

# Layers to fine-tune
unfreeze_layers:
  - "classifier"
  - "pooler"
  - "layer_11"
```

Each entry under `training_sets` and `test_sets` is a named dataset with:
- `classification_system`: one of `accessory_components`, `alteration_degree`, `alteration_degree_consolidated`, `alteration_degree_unconsolidated`, `cementation`, `color`, `color_consolidated`, `color_unconsolidated`, `debris`, `en_main`, `grain_angularity`, `grain_shape`, `lithology`, `mineral_components`, `organic_components`, or `uscs`
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

## 4. Train on Azure ML (remote GPU cluster)

For long runs, submit the job to the Azure ML cluster instead of running locally.

1. **Register a ground-truth folder asset** in Azure ML Studio containing all ground-truth JSON files (uri-folder). Name it `ground-truth-data`. The filenames must match those referenced in your BERT config YAML.

2. **Authenticate**:
   ```bash
   az login
   ```

3. **Submit a job**:

```bash
python src/scripts/azure_jobs/run_azure_train_bert.py -cf {config_file_path}
```

The script builds (or reuses) the Docker environment, uploads the `src/` code snapshot, and submits the job. It prints the run name and a Studio URL to monitor progress.

### How it works

The job mounts the `ground-truth-data` folder asset and sets `BOREHOLES_DATA_PATH` to its mount path, then calls the training module. It is the same entrypoint used locally (as with `fine-tune-bert`).


## 5. Results

| Dataset                            | Support (num classes) | Target | F1-macro | F1-micro |
|------------------------------------|-----------------------|--------|----------|----------|
| `accessory_components`             |               75 (66) |  Multi |        - |        - |
| `alteration_degree_consolidated`   |             2,716 (8) | Single |        - |        - |
| `alteration_degree_unconsolidated` |                41 (8) | Single |        - |        - |
| `cementation`                      |                78 (7) | Single |        - |        - |
| `color_consolidated`*              |           16,143 (91) | Single |    0.487 |    0.752 |
| `color_unconsolidated`*            |           20,121 (91) | Single |    0.502 |    0.803 |
| `debris`                           |            70,084 (7) |  Multi |        - |        - |
| `en_main`                          |           89,842 (34) | Single |    0.xxx |    0.xxx |
| `grain_angularity`                 |            70,377 (8) |  Multi |    0.831 |    0.974 |
| `grain_shape`                      |            70,087 (5) |  Multi |        - |        - |
| `lithology`                        |           45,323 (61) | Single |    0.848 |    0.942 |
| `mineral_components`               |              63 (111) |  Multi |        - |        - |
| `organic_components`               |           70,102 (11) |  Multi |        - |        - |
| `uscs`                             |            9,917 (38) | Single |    0.329 |    0.602 |

* Model jointly trained, same for both tasks.
