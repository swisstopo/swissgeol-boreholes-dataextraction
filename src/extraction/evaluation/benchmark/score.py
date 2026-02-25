"""Evaluate the predictions against the ground truth."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from core.mlflow_tracking import mlflow
from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from swissgeol_doc_processing.utils.file_utils import get_data_path

load_dotenv()

logger = logging.getLogger(__name__)


class ExtractionBenchmarkSummary(BaseModel):
    """Helper class containing a summary of all the results of a single benchmark."""

    ground_truth_path: str
    n_documents: int
    geology: dict[str, float]
    metadata: dict[str, float]

    def metrics(self, short: bool = False) -> dict[str, float]:
        def key(category: str, metric: str) -> str:
            if short:
                return metric
            else:
                return f"{category}/{metric}"

        geology_dict = {key("geology", metric): value for metric, value in self.geology.items()}
        metadata_dict = {key("metadata", metric): value for metric, value in self.metadata.items()}
        return geology_dict | metadata_dict


def evaluate_all_predictions(
    predictions: OverallFilePredictions,
    ground_truth_path: Path,
) -> None | ExtractionBenchmarkSummary:
    """Computes all the metrics, logs them, and creates corresponding MLFlow artifacts (when enabled).

    Args:
        predictions (OverallFilePredictions): The predictions objects.
        ground_truth_path (Path | None): The path to the ground truth file.

    Returns:
        ExtractionBenchmarkSummary | None: A JSON-serializable ExtractionBenchmarkSummary
        that can be used by multi-benchmark runners.
    """
    if not (ground_truth_path and ground_truth_path.exists()):  # for inference no ground truth is available
        logger.warning("Ground truth file not found. Skipping evaluation.")
        return None

    ground_truth = GroundTruth(ground_truth_path)

    #############################
    # Evaluate the borehole extraction
    #############################
    matched_with_ground_truth = predictions.match_with_ground_truth(ground_truth)
    metrics = matched_with_ground_truth.evaluate_geology()
    metrics_dict = metrics.metrics_dict()

    # Format the metrics dictionary to limit to three digits
    formatted_metrics = {k: f"{v:.3f}" for k, v in metrics_dict.items()}
    logger.info("Performance metrics: %s", formatted_metrics)

    #############################
    # Evaluate the borehole extraction metadata
    #############################

    metadata_metrics_list = matched_with_ground_truth.evaluate_metadata_extraction()
    metadata_metrics = metadata_metrics_list.get_cumulated_metrics()
    document_level_metadata_metrics: pd.DataFrame = metadata_metrics_list.get_document_level_metrics()

    # print the metrics
    logger.info("Metadata Performance metrics:")
    logger.info(metadata_metrics.to_json())

    if mlflow:
        mlflow.log_metrics(metrics_dict)
        mlflow.log_metrics(metadata_metrics.to_json())

        # # Create temporary folder to dump csv file and track them using MLFlow
        with tempfile.TemporaryDirectory() as temp_directory:
            document_level_metadata_metrics.to_csv(
                Path(temp_directory) / "document_level_metadata_metrics.csv", index_label="document_name"
            )  # mlflow.log_artifact expects a file
            metrics.document_level_metrics_df().to_csv(
                Path(temp_directory) / "document_level_metrics.csv", index_label="document_name"
            )  # mlflow.log_artifact expects a file
            mlflow.log_artifact(Path(temp_directory) / "document_level_metrics.csv")
            mlflow.log_artifact(Path(temp_directory) / "document_level_metadata_metrics.csv")

    return ExtractionBenchmarkSummary(
        ground_truth_path=str(ground_truth_path),
        n_documents=len(predictions.file_predictions_list),
        geology=metrics_dict,
        metadata=metadata_metrics.to_json(),
    )


def main():
    """Main function to evaluate the predictions against the ground truth."""
    args = parse_cli()

    # Load the predictions
    try:
        with open(args.predictions_path, encoding="utf8") as file:
            predictions = json.load(file)
    except FileNotFoundError:
        logger.error("Predictions file not found: %s", args.predictions_path)
        return
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from predictions file: %s", e)
        return

    predictions = OverallFilePredictions.from_json(predictions)
    if mlflow:
        mlflow.set_experiment("Boreholes Stratigraphy")
        with mlflow.start_run():
            evaluate_all_predictions(predictions, args.ground_truth_path)
    else:
        evaluate_all_predictions(predictions, args.ground_truth_path)


def parse_cli() -> argparse.Namespace:
    """Parse the command line arguments and pass them to the main function."""
    parser = argparse.ArgumentParser(description="Borehole Stratigraphy Evaluation Script")

    # Add arguments with defaults
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=get_data_path().parent / "data" / "zurich_ground_truth.json",
        help="Path to the ground truth JSON file (default: '../data/zurich_ground_truth.json').",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=get_data_path() / "output" / "predictions.json",
        help="Path to the predictions JSON file (default: './output/predictions.json').",
    )

    # Parse arguments and pass to main
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and pass to main
    main()
