"""Submit a BERT fine-tuning job to Azure ML.

The only line to change between runs is CONFIG_FILE. The script reads the
BERT config YAML, discovers which ground-truth datasets are required, and
loads them from the Azure ML registered data assets automatically.

Convention: each ground-truth filename in the config (e.g. deepwells_ground_truth.json)
must be registered in Azure ML with the matching asset name (deepwells_ground_truth).

Submit the job with: python src/scripts/azure_jobs/job_train_bert.py
"""

import os
from pathlib import Path

import yaml
from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import BuildContext, Environment
from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Only this line needs to change between runs
CONFIG_FILE = "bert/bert_config_cementation.yml"
# =============================================================================

# Parse the BERT config to discover required ground-truth datasets
config_path = Path("src/classification/config") / CONFIG_FILE
with open(config_path) as f:
    config = yaml.safe_load(f)

gt_filenames = {
    gt
    for dataset in (list(config.get("training_sets", {}).values()) + list(config.get("test_sets", {}).values()))
    for gt in dataset.get("ground_truths", [])
}

# Map filename stem → Azure ML registered asset (version 1)
# e.g. deepwells_ground_truth.json → azureml:deepwells_ground_truth:1
job_inputs = {Path(gt).stem: Input(type=AssetTypes.URI_FILE, path=f"azureml:{Path(gt).stem}:1") for gt in gt_filenames}

# Copy each asset to /tmp/gt_data/ preserving the filename from the config
copy_commands = " && ".join(f"cp '${{{{inputs.{Path(gt).stem}}}}}' /tmp/gt_data/{gt}" for gt in gt_filenames)

job_command = (
    "pip install -e . --no-deps && "
    "mkdir -p /tmp/gt_data && "
    f"{copy_commands} && "
    "export BOREHOLES_DATA_PATH=/tmp/gt_data && "
    f"python src/scripts/azure_jobs/run_bert_training.py -cf {CONFIG_FILE}"
)

job_env_vars = {"MLFLOW_TRACKING": "True"}

# Connect to your workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
)

env = Environment(
    name="bert-training-env",
    version="3",
    build=BuildContext(
        path="src/scripts/azure_jobs",
        dockerfile_path="Dockerfile.azureml",
    ),
    description="Pre-built environment for BERT training (deep-learning + experiment-tracking extras)",
)

try:
    registered_env = ml_client.environments.create_or_update(env)
except ResourceExistsError:
    registered_env = ml_client.environments.get(name=env.name, version=env.version)

job = command(
    code=".",
    command=job_command,
    compute=os.environ["AZURE_COMPUTE_NAME"],
    environment=f"{registered_env.name}:{registered_env.version}",
    experiment_name=config.get("experiment_name", Path(CONFIG_FILE).stem),
    inputs=job_inputs,
    environment_variables=job_env_vars,
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Monitor at: {returned_job.studio_url}")
