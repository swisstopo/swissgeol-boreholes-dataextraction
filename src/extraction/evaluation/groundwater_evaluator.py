"""Classes for evaluating the groundwater levels of a borehole."""

from dataclasses import dataclass

from extraction.evaluation.benchmark.metrics import OverallMetrics
from extraction.evaluation.evaluation_dataclasses import Metrics
from extraction.evaluation.utility import evaluate
from extraction.features.groundwater.groundwater_extraction import Groundwater
from extraction.features.predictions.borehole_predictions import FileGroundwaterWithGroundTruth


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
        """Convert the overall groundwater metrics to an OverallMetrics object."""
        overall_metrics = OverallMetrics()
        for groundwater_metrics in self.groundwater_metrics:
            overall_metrics.metrics[groundwater_metrics.filename] = groundwater_metrics.groundwater_metrics
        return overall_metrics

    def groundwater_depth_metrics_to_overall_metrics(self):
        """Convert the overall groundwater depth metrics to an OverallMetrics object."""
        overall_metrics = OverallMetrics()
        for groundwater_metrics in self.groundwater_metrics:
            overall_metrics.metrics[groundwater_metrics.filename] = groundwater_metrics.groundwater_depth_metrics
        return overall_metrics


class GroundwaterEvaluator:
    """Class for evaluating the extracted groundwater information of a borehole."""

    def __init__(
        self,
        groundwater_list: list[FileGroundwaterWithGroundTruth],
    ):
        """Initializes the GroundwaterEvaluator object.

        Args:
            groundwater_list (list[FileGroundwaterWithGroundTruth]): A list of extracted groundwater data per
                borehole, together with the ground truth data associated to it.
        """
        self.groundwater_list = groundwater_list

    def evaluate(self) -> OverallGroundwaterMetrics:
        """Evaluate the groundwater information of the file against the ground truth.

        Returns:
            OverallGroundwaterMetrics: The overall groundwater metrics.
        """
        overall_groundwater_metrics = OverallGroundwaterMetrics()

        for file in self.groundwater_list:
            # lists to contain the metrics
            groundwater_metrics_list = []
            groundwater_depth_metrics_list = []
            groundwater_elevation_metrics_list = []
            groundwater_date_metrics_list = []

            for borehole_data in file.boreholes:
                ############################################################################################################
                ### Compute the metadata correctness for the groundwater information.
                ############################################################################################################
                gt_groundwater = [
                    Groundwater.from_json_values(
                        depth=json_gt_data.get("depth"),
                        date=json_gt_data.get("date"),
                        elevation=json_gt_data.get("elevation"),
                    )
                    for json_gt_data in borehole_data.ground_truth
                ]

                entries = (
                    [feature_on_page.feature for feature_on_page in borehole_data.groundwater.groundwater_feature_list]
                    if borehole_data.groundwater
                    else []
                )

                groundwater_evaluation = evaluate(entries, gt_groundwater, self.match_groundwater)
                groundwater_metrics_list.append(groundwater_evaluation.metrics)
                for feature, is_correct in zip(entries, groundwater_evaluation.extracted_correct, strict=True):
                    feature.is_correct = is_correct

                groundwater_depth_metrics = evaluate(entries, gt_groundwater, self.match_groundwater_depth).metrics
                groundwater_depth_metrics_list.append(groundwater_depth_metrics)

                groundwater_elevation_metrics = evaluate(
                    entries, gt_groundwater, self.match_groundwater_elevation
                ).metrics
                groundwater_elevation_metrics_list.append(groundwater_elevation_metrics)

                groundwater_date_metrics = evaluate(entries, gt_groundwater, self.match_groundwater_date).metrics
                groundwater_date_metrics_list.append(groundwater_date_metrics)

            # we take the micro-average across boreholes
            file_groundwater_metrics = GroundwaterMetrics(
                groundwater_metrics=Metrics.micro_average(groundwater_metrics_list),
                groundwater_depth_metrics=Metrics.micro_average(groundwater_depth_metrics_list),
                groundwater_elevation_metrics=Metrics.micro_average(groundwater_elevation_metrics_list),
                groundwater_date_metrics=Metrics.micro_average(groundwater_date_metrics_list),
                filename=file.filename,
            )

            overall_groundwater_metrics.add_groundwater_metrics(file_groundwater_metrics)

        return overall_groundwater_metrics

    @staticmethod
    def match_groundwater(extracted: Groundwater, ground_truth: Groundwater) -> bool:
        extracted_tuple = (extracted.depth, extracted.format_date(), extracted.elevation)
        ground_truth_tuple = (ground_truth.depth, ground_truth.format_date(), ground_truth.elevation)
        return extracted_tuple == ground_truth_tuple

    @staticmethod
    def match_groundwater_depth(extracted: Groundwater, ground_truth: Groundwater) -> bool:
        return extracted.depth == ground_truth.depth

    @staticmethod
    def match_groundwater_date(extracted: Groundwater, ground_truth: Groundwater) -> bool:
        return extracted.format_date() == ground_truth.format_date()

    @staticmethod
    def match_groundwater_elevation(extracted: Groundwater, ground_truth: Groundwater) -> bool:
        return extracted.elevation == ground_truth.elevation
