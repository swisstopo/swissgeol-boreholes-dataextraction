"""Classes for evaluating the groundwater levels of a borehole."""

from dataclasses import dataclass

from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.benchmark.metrics import OverallMetrics
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

    def __init__(self):
        self.groundwater_metrics: list[GroundwaterMetrics] = []

    def add_groundwater_metrics(self, groundwater_metrics: GroundwaterMetrics):
        """Add groundwater metrics to the list.

        Args:
            groundwater_metrics (GroundwaterMetrics): The groundwater metrics to add.
        """
        self.groundwater_metrics.append(groundwater_metrics)

    def groundwater_metrics_to_overall_metrics(self):
        """Convert the overall groundwater metrics to a OverallMetrics object."""
        overall_metrics = OverallMetrics()
        for groundwater_metrics in self.groundwater_metrics:
            overall_metrics.metrics[groundwater_metrics.filename] = groundwater_metrics.groundwater_metrics
        return overall_metrics

    def groundwater_depth_metrics_to_overall_metrics(self):
        """Convert the overall groundwater depth metrics to a OverallMetrics object."""
        overall_metrics = OverallMetrics()
        for groundwater_metrics in self.groundwater_metrics:
            overall_metrics.metrics[groundwater_metrics.filename] = groundwater_metrics.groundwater_depth_metrics
        return overall_metrics


class GroundwaterEvaluator:
    """Class for evaluating the extracted groundwater information of a borehole."""

    def __init__(self, groundwater_entries: list[GroundwaterInDocument], ground_truth_path: str):
        """Initializes the GroundwaterEvaluator object.

        Args:
            groundwater_entries (list[GroundwaterInDocument]): The metadata to evaluate.
            ground_truth_path (str): The path to the ground truth file.
        """
        # Load the ground truth data for the metadata
        self.groundwater_ground_truth = GroundTruth(ground_truth_path)
        self.groundwater_entries: list[GroundwaterInDocument] = groundwater_entries

    def evaluate(self) -> OverallGroundwaterMetrics:
        """Evaluate the groundwater information of the file against the ground truth.

        Returns:
            OverallGroundwaterMetrics: The overall groundwater metrics.
        """
        overall_groundwater_metrics = OverallGroundwaterMetrics()

        for groundwater_in_doc in self.groundwater_entries:
            filename = groundwater_in_doc.filename
            ground_truth_data = self.groundwater_ground_truth.for_file(filename)
            if ground_truth_data is None or ground_truth_data.get("groundwater") is None:
                ground_truth = []  # If no ground truth is available, set it to an empty list
            else:
                ground_truth = ground_truth_data.get("groundwater")

            ############################################################################################################
            ### Compute the metadata correctness for the groundwater information.
            ############################################################################################################
            gt_groundwater = [
                Groundwater.from_json_values(
                    depth=json_gt_data.get("depth"),
                    date=json_gt_data.get("date"),
                    elevation=json_gt_data.get("elevation"),
                )
                for json_gt_data in ground_truth
            ]

            groundwater_metrics = count_against_ground_truth(
                [
                    (
                        entry.depth,
                        entry.format_date(),
                        entry.elevation,
                    )
                    for entry in groundwater_in_doc.groundwater
                ],
                [(entry.depth, entry.format_date(), entry.elevation) for entry in gt_groundwater],
            )
            groundwater_depth_metrics = count_against_ground_truth(
                [entry.depth for entry in groundwater_in_doc.groundwater],
                [entry.depth for entry in gt_groundwater],
            )
            groundwater_elevation_metrics = count_against_ground_truth(
                [entry.elevation for entry in groundwater_in_doc.groundwater],
                [entry.elevation for entry in gt_groundwater],
            )
            groundwater_date_metrics = count_against_ground_truth(
                [entry.format_date() for entry in groundwater_in_doc.groundwater],
                [entry.format_date() for entry in gt_groundwater],
            )

            file_groundwater_metrics = GroundwaterMetrics(
                groundwater_metrics=groundwater_metrics,
                groundwater_depth_metrics=groundwater_depth_metrics,
                groundwater_elevation_metrics=groundwater_elevation_metrics,
                groundwater_date_metrics=groundwater_date_metrics,
                filename=filename,
            )  # TODO: This clashes with the OverallMetrics object

            overall_groundwater_metrics.add_groundwater_metrics(file_groundwater_metrics)

        return overall_groundwater_metrics
