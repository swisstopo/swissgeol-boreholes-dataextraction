# Train BERT Model  

To fine-tune BERT on your data, follow these steps:  

## 1. Setup  

Repeat steps 1 and 2 of the [data extraction pipeline](#run-data-extraction) to set up the environment and download the data.  

## 2. Prepare Data 
Training requires a JSON file containing layer descriptions and their ground truth classes.
Each file contains boreholes, and each borehole contains layers.
For each layer:
- `material_description` - input text for BERT
- a classification label depending on the system:
    - `uscs_1` for USCS
    - `unconsolidated`(main/other) for EN
    - `lithology`
    
For each system, you should create an own config file, e.g. `bert_config_uscs.yml` which defines training data as well as model parameters. 
It could look like this: 
```yml
# model_path: "google-bert/bert-base-uncased"
model_path: "google-bert/bert-base-multilingual-uncased"

# Data parameters
classification_system: "en_main"
json_file_name: "geoquat_ground_truth.json"
train_subset: "geoquat/train"
eval_subset: "geoquat/validation"

# Training hyperparameters
batch_size: 32
num_epochs: 32
learning_rate: 1e-4 # 3e-3 for stage 1, 1e-4 for stage 2
weight_decay: 0.001
warmup_ratio: 0.1
lr_scheduler_type: "cosine_with_restarts"
max_grad_norm: 5.0
# Custom Training hyperparameters
unfreeze_layers:
  - "classifier"
  - "pooler"
  - "layer_11" # uncomment for stage 2
use_class_balancing: "false"

# Inference parameters
inference_batch_size: 32
```
Layers without valid ground truth are automatically skipped.

## 3. Choose Hyperparameters  

Modify the file `config/bert_config_uscs.yml` to set the hyperparameters for training and data processing. Data sources used for training and validation are specified in this file.

## 4. Train the Model  

To fine-tune BERT from the base model on Hugging Face, run:  

```bash
fine-tune-bert -cf bert_config_uscs.yml
```  

- Use `-cf` or `--config-file-path` to specify the config file containing the training parameters.
- By default, the initial model is the one specified in the config file (loaded from hugging face). However, you can continue training from a specific checkpoint by providing a local model path with `-c` or `--model-checkpoint`.  

The pipeline stores a checkpoint of the model after each epoch and logs training details in the `models` directory. The model name corresponds to the timestamp when training was launched.  

