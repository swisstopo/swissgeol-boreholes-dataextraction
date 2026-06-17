"""Entrypoint for BERT fine-tuning — works locally and as an Azure ML job command.

Run locally (BOREHOLES_DATA_PATH must point to your ground-truth directory, or
it defaults to data/ at the repo root):

    python src/scripts/azure_jobs/run_bert_training.py -cf bert/bert_config_uscs.yml

Submit to Azure ML (handled by job_train_bert.py, which mounts the registered
ground-truth folder and sets BOREHOLES_DATA_PATH before calling this script):

    python src/scripts/azure_jobs/job_train_bert.py
"""

from classification.models.train import train_model

if __name__ == "__main__":
    train_model()
