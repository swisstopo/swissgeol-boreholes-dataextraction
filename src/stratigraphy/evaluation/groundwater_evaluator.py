"""Classes for evaluating the groundwater levels of a borehole."""

from dataclasses import dataclass
from typing import Any

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import DatasetMetrics
from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.evaluation.utility import count_against_ground_truth
from stratigraphy.groundwater.groundwater_extraction import Groundwater, GroundwaterInDocument


@dataclass
class GroundwaterMetrics:
    """Class for storing the metrics of the groundwater information."""

    groundwater_metrics: Metrics = None
    groundwater_depth_metrics: Metrics = None
    groundwater_elevation_metrics: Metrics = None
    groundwater_date_metrics: Metrics = None
    filename: str = None


class OverallGroundwaterMetrics:
    """Class for storing the overall metrics of the groundwater information."""

    groundwater_metrics: list[GroundwaterMetrics] = None

    def __init__(self):
        self.groundwater_metrics = []

    def add_groundwater_metrics(self, groundwater_metrics: GroundwaterMetrics):
        """Add groundwater metrics to the list.

        Args:
            groundwater_metrics (GroundwaterMetrics): The groundwater metrics to add.
        """
        self.groundwater_metrics.append(groundwater_metrics)

    def groundwater_metrics_to_dataset_metrics(self):
        """Convert the overall groundwater metrics to a DatasetMetrics object."""
        dataset_metrics = DatasetMetrics()
        for groundwater_metrics in self.groundwater_metrics:
            dataset_metrics.metrics[groundwater_metrics.filename] = groundwater_metrics.groundwater_metrics
        return dataset_metrics

    def groundwater_depth_metrics_to_dataset_metrics(self):
        """Convert the overall groundwater depth metrics to a DatasetMetrics object."""
        dataset_metrics = DatasetMetrics()
        for groundwater_metrics in self.groundwater_metrics:
            dataset_metrics.metrics[groundwater_metrics.filename] = groundwater_metrics.groundwater_depth_metrics
        return dataset_metrics


class GroundwaterEvaluator:
    """Class for evaluating the extracted groundwater information of a borehole."""

    groundwater_entries: list[GroundwaterInDocument] = None
    groundwater_ground_truth: dict[str, Any] = None

    def __init__(self, groundwater_entries: list[GroundwaterInDocument], ground_truth_path: str):
        """Initializes the GroundwaterEvaluator object.

        Args:
            groundwater_entries (list[GroundwaterInDocument]): The metadata to evaluate.
            ground_truth_path (str): The path to the ground truth file.
        """
        # Load the ground truth data for the metadata
        self.groundwater_ground_truth = GroundTruth(ground_truth_path)
        if self.groundwater_ground_truth is None:
            self.groundwater_ground_truth = []

        self.groundwater_entries = groundwater_entries

    def evaluate(self):
        """Evaluate the groundwater information of the file against the ground truth.

        Args:
            groundwater_ground_truth (list): The ground truth for the file.
        """
        overall_groundwater_metrics = OverallGroundwaterMetrics()

        for groundwater_in_doc in self.groundwater_entries:
            filename = groundwater_in_doc.filename
            ground_truth = self.groundwater_ground_truth.for_file(filename).get("groundwater", [])
            if ground_truth is None:
                ground_truth = []  # If no ground truth is available, set it to an empty list

            ############################################################################################################
            ### Compute the metadata correctness for the groundwater information.
            ############################################################################################################
            gt_groundwater = [
                Groundwater.from_json_values(
                    depth=json_gt_data["depth"],
                    date=json_gt_data["date"],
                    elevation=json_gt_data["elevation"],
                )
                for json_gt_data in ground_truth
            ]

            groundwater_metrics = count_against_ground_truth(
                [
                    (
                        entry.groundwater.depth,
                        entry.groundwater.format_date(),
                        entry.groundwater.elevation,
                    )
                    for entry in groundwater_in_doc.groundwater
                ],
                [(entry.depth, entry.format_date(), entry.elevation) for entry in gt_groundwater],
            )
            groundwater_depth_metrics = count_against_ground_truth(
                [entry.groundwater.depth for entry in groundwater_in_doc.groundwater],
                [entry.depth for entry in gt_groundwater],
            )
            groundwater_elevation_metrics = count_against_ground_truth(
                [entry.groundwater.elevation for entry in groundwater_in_doc.groundwater],
                [entry.elevation for entry in gt_groundwater],
            )
            groundwater_date_metrics = count_against_ground_truth(
                [entry.groundwater.date for entry in groundwater_in_doc.groundwater],
                [entry.date for entry in gt_groundwater],
            )

            file_groundwater_metrics = GroundwaterMetrics(
                groundwater_metrics=groundwater_metrics,
                groundwater_depth_metrics=groundwater_depth_metrics,
                groundwater_elevation_metrics=groundwater_elevation_metrics,
                groundwater_date_metrics=groundwater_date_metrics,
                filename=filename,
            )  # TODO: This clashes with the DatasetMetrics object

            overall_groundwater_metrics.add_groundwater_metrics(file_groundwater_metrics)

        return overall_groundwater_metrics
