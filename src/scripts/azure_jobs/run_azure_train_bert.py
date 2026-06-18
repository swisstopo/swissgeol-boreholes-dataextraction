"""Submit a BERT fine-tuning job to Azure ML.

The script reads the BERT config YAML, discovers which ground-truth datasets
are required, and loads them from the Azure ML registered data assets automatically.

Convention: each ground-truth filename in the config (e.g. deepwells_ground_truth.json)
must be registered in Azure ML with the matching asset name (deepwells_ground_truth).

Submit the job with:
    python src/scripts/azure_jobs/run_azure_train_bert.py --config bert/bert_config_cementation.yml
"""

import logging
import os

import click
from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import BuildContext, Environment
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from classification.models.config import ExperimentConfig
from classification.utils.file_utils import read_params

logger = logging.getLogger(__name__)

load_dotenv()


def get_environement(ml_client: MLClient) -> Environment:
    """Register or retrieve the BERT training environment.

    Builds the environment from the project Dockerfile if version does not exist yet,
    otherwise returns the already-registered version without rebuilding.

    Args:
        ml_client (MLClient): Authenticated Azure ML client for the target workspace.

    Returns:
        Environment: The registered Azure ML Environment entity.
    """
    env = Environment(
        name="bert-training-env",
        version="9",
        build=BuildContext(
            path=".",
            dockerfile_path="src/scripts/azure_jobs/Dockerfile.azureml",
        ),
        description="Pre-built environment for BERT training (deep-learning + experiment-tracking extras)",
    )

    try:
        return ml_client.environments.create_or_update(env)
    except ResourceExistsError:
        return ml_client.environments.get(name=env.name, version=env.version)


@click.command()
@click.option(
    "-cf",
    "--config-file-path",
    required=True,
    type=str,
    help="Name (not path) of the configuration yml file inside the `config` folder.",
)
@click.option(
    "-u",
    "--uri-folder",
    type=str,
    default="ground-truth-data",
    help="Name of the data folder containing ground truths.",
)
def run(config_file_path: str, uri_folder: str) -> None:
    """Submit a BERT fine-tuning job to Azure ML.

    Connects to the Azure ML workspace, ensures the training environment is registered,
    resolves the ground-truth data asset by name, and submits a command job that runs
    `train.py` on the specified compute cluster.

    Required environment variables (loaded from .env):
        AZURE_SUBSCRIPTION_ID: Azure subscription ID.
        AZURE_RESOURCE_GROUP: Resource group containing the workspace.
        AZURE_WORKSPACE_NAME: Azure ML workspace name.
        AZURE_COMPUTE_NAME: Name of the compute cluster to run the job on.

    Args:
        config_file_path (str): Path to the BERT config YAML file (relative to repo root).
        uri_folder (str): Name of the registered Azure ML data asset (URI_FOLDER) holding
            the ground-truth JSON files. Must already exist in the workspace.

    Example:
        ```bash
        python src/scripts/azure_jobs/run_azure_train_bert.py -cf bert/bert_config_cementation.yml
        ```
    """
    # Load BERT experiment configuration
    model_config = ExperimentConfig.model_validate(read_params(config_file_path))

    # Connect to your workspace
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )

    # Get or update ressources
    registered_env = get_environement(ml_client)

    try:
        data = ml_client.data.get(name=uri_folder, label="latest")
    except ResourceNotFoundError:
        logger.error(f"Azure: unknown ressource {uri_folder=}")
        exit()

    job = command(
        # Copy all files (execpt .amlignore)
        code=".",
        # Set data path as mounted disk on Azure and run training
        command=(
            "export BOREHOLES_DATA_PATH='${{inputs.gt_data}}'"
            f" && python src/classification/models/train.py -cf {config_file_path}"
        ),
        # Compute instance name
        compute=os.environ["AZURE_COMPUTE_NAME"],
        # Predefined running env
        environment=f"{registered_env.name}:{registered_env.version}",
        experiment_name=model_config.experiment_name,
        # Input data to mount with job
        inputs={"gt_data": Input(type=AssetTypes.URI_FOLDER, path=f"azureml:{data.name}:{data.version}")},
        # Force MLflow tracking and set src as package root
        environment_variables={
            "MLFLOW_TRACKING": "True",
            "PYTHONPATH": "src",
        },
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted: {returned_job.name}")
    print(f"Monitor at: {returned_job.studio_url}")


if __name__ == "__main__":
    run()
