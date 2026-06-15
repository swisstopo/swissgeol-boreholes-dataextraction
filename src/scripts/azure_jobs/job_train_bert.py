"""Submit a BERT fine-tuning job to Azure ML."""

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import DefaultAzureCredential

# Connect to your workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="f12e214d-46c6-49bf-a083-f89cf9c3179d",
    resource_group_name="rg-swisstopo-compute",
    workspace_name="aml-swisstopo-sn",
    # compute="swisstopo-boreholes",
    compute="swisstopo-dev",
)

# =============================================================================
# Data via registered data asset pointer (URI_FILE)
# Requires: workspace managed identity has "Storage Blob Data Reader" on asaswisstopoaml
# Each file must be registered as a data asset in Azure ML Studio
# =============================================================================

job_inputs = {
    "deepwells": Input(type=AssetTypes.URI_FILE, path="azureml:deepwells_ground_truth:1"),
    "geoquat": Input(type=AssetTypes.URI_FILE, path="azureml:geoquat_ground_truth:1"),
}
job_command = (
    "pip install -e . --no-deps && "
    "mkdir -p /tmp/gt_data && "
    "cp '${{inputs.deepwells}}' /tmp/gt_data/ && "
    "cp '${{inputs.geoquat}}' /tmp/gt_data/ && "
    "export BOREHOLES_DATA_PATH=/tmp/gt_data && "
    "python -m src.classification.models.train -cf bert/bert_config_cementation.yml"
    # "python -m src.classification.models.train -cf bert/test_bert_config_debris.yml"
)

job_env_vars = {"MLFLOW_TRACKING": "True"}

# =============================================================================

env = Environment(
    name="bert-training-env",
    version="3",
    build=BuildContext(
        path="src/scripts/azure_jobs",
        dockerfile_path="Dockerfile.azureml",
    ),
    description="Pre-built environment for BERT training (deep-learning + experiment-tracking extras)",
)

registered_env = ml_client.environments.create_or_update(env)

job = command(
    code=".",
    command=job_command,
    compute="swisstopo-dev",
    environment=f"{registered_env.name}:{registered_env.version}",
    experiment_name="test-bert-training-cementation",
    inputs=job_inputs,
    environment_variables=job_env_vars,
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Monitor at: {returned_job.studio_url}")

# Submit the job with: python src/scripts/azure_jobs/job_train_bert.py
