"""Evaluate the predictions against the ground truth."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from core.benchmark_utils import BenchmarkSummary
from core.mlflow_tracking import mlflow
from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.evaluation.evaluator import Evaluator
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from swissgeol_doc_processing.utils.file_utils import get_data_path

load_dotenv()

logger = logging.getLogger(__name__)


class ExtractionBenchmarkSummary(BenchmarkSummary):
    """Summary of a single extraction benchmark."""

    geology: dict[str, float]
    metadata: dict[str, float]

    def metrics_flat(self, short: bool = False) -> dict[str, float]:
        def key(category: str, metric: str) -> str:
            if short:
                return metric
            return f"{category}/{metric}"

        geology_dict = {key("geology", metric): value for metric, value in self.geology.items()}
        metadata_dict = {key("metadata", metric): value for metric, value in self.metadata.items()}
        return geology_dict | metadata_dict


def evaluate_prediction(
    prediction: FilePredictions,
    ground_truth: GroundTruth | None = None,
) -> FilePredictions:
    """Computes metrics for a given file.

    Note that evaluation mutates the `is_correct` flags on layers, metadata,
    and groundwater entries inside `prediction`.

    Args:
        prediction (FilePredictions): The predictions object.
        ground_truth (GroundTruth | None): The ground truth object. If None, no evaluation is performed.

    Returns:
        FilePredictions: The prediction, with evaluation flags set when ground truth is provided.
    """
    if ground_truth is None:
        return prediction

    # Create prediction with GT
    matched_with_gt = Evaluator.match_with_ground_truth(prediction, ground_truth)

    # Run evaluation for file (! mutates prediction !)
    Evaluator.evaluate(matched_with_gt)

    return prediction


def evaluate_all_predictions(
    predictions: OverallFilePredictions,
    ground_truth: GroundTruth | None = None,
) -> None | ExtractionBenchmarkSummary:
    """Computes all the metrics, logs them, and creates corresponding MLFlow artifacts (when enabled).

    Args:
        predictions (OverallFilePredictions): The predictions objects.
        ground_truth (GroundTruth | None): The ground truth object.

    Returns:
        ExtractionBenchmarkSummary | None: A JSON-serializable ExtractionBenchmarkSummary
        that can be used by multi-benchmark runners.
    """
    if ground_truth is None:
        return None

    matched_list_with_gt = Evaluator.match_overall_with_ground_truth(predictions, ground_truth)

    # Evaluate overall extraction
    geology_metrics, metadata_metrics_list = Evaluator.evaluate_overall(matched_list_with_gt)

    logger.info("Macro avg:")
    logger.info(
        "layer f1: %.1f%%, depth interval f1: %.1f%%, material description f1: %.1f%%",
        geology_metrics.layer_metrics.macro_f1() * 100,
        geology_metrics.depth_interval_metrics.macro_f1() * 100,
        geology_metrics.material_description_metrics.macro_f1() * 100,
    )

    # Log detailed geology metrics
    geology_metrics_dict = geology_metrics.metrics_dict()
    logger.info("Performance metrics: %s", {k: f"{v:.3f}" for k, v in geology_metrics_dict.items()})

    # Log detailed metadata metrics
    metadata_metrics = metadata_metrics_list.get_cumulated_metrics()
    document_level_metadata_metrics: pd.DataFrame = metadata_metrics_list.get_document_level_metrics()
    logger.info("Metadata Performance metrics: %s", metadata_metrics.to_json())

    # Log results
    if mlflow:
        mlflow.log_metrics(geology_metrics_dict)
        mlflow.log_metrics(metadata_metrics.to_json())

        # Create temporary folder to dump csv file and track them using MLFlow
        with tempfile.TemporaryDirectory() as temp_directory:
            # Metadata
            document_level_metadata_metrics.to_csv(
                Path(temp_directory) / "document_level_metadata_metrics.csv", index_label="document_name"
            )  # mlflow.log_artifact expects a file
            mlflow.log_artifact(Path(temp_directory) / "document_level_metadata_metrics.csv")
            # Geology
            geology_metrics.document_level_metrics_df().to_csv(
                Path(temp_directory) / "document_level_metrics.csv", index_label="document_name"
            )  # mlflow.log_artifact expects a file
            mlflow.log_artifact(Path(temp_directory) / "document_level_metrics.csv")

    return ExtractionBenchmarkSummary(
        ground_truth_path=str(ground_truth.path),
        n_documents=len(predictions.file_predictions_list),
        geology=geology_metrics_dict,
        metadata=metadata_metrics.to_json(),
    )


def main():
    """Main function to evaluate the predictions against the ground truth."""
    args = parse_cli()

    # Load the predictions
    try:
        with open(args.predictions_path, encoding="utf8") as file:
            predictions = json.load(file)
        predictions = OverallFilePredictions.from_json(predictions)
    except FileNotFoundError:
        logger.error("Predictions file not found: %s", args.predictions_path)
        return
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from predictions file: %s", e)
        return

    # Load ground truth
    try:
        ground_truth = GroundTruth(args.ground_truth_path)
    except FileNotFoundError:
        logger.error("Ground truth file not found: %s", args.ground_truth_path)
        return

    if mlflow:
        mlflow.set_experiment("Boreholes Stratigraphy")
        with mlflow.start_run():
            evaluate_all_predictions(predictions, ground_truth)
    else:
        evaluate_all_predictions(predictions, ground_truth)


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
