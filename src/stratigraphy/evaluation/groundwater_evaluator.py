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
        groundwater_entries: list[GroundwaterInDocument],
        ground_truth: GroundTruth,
        gt_to_pred_matching: dict[str : dict[int, int]],
    ):
        """Initializes the GroundwaterEvaluator object.

        Args:
            groundwater_entries (list[GroundwaterInDocument]): The metadata to evaluate.
            ground_truth (GroundTruth): The ground truth.
            gt_to_pred_matching (dict[str : dict[int:int]]): the dict matching the index of the gt borehole to pred
        """
        self.ground_truth = ground_truth
        self.groundwater_entries: list[GroundwaterInDocument] = groundwater_entries
        self.gt_to_pred_matching: dict[str : dict[int:int]] = gt_to_pred_matching

    def evaluate(self) -> OverallGroundwaterMetrics:
        """Evaluate the groundwater information of the file against the ground truth.

        Returns:
            OverallGroundwaterMetrics: The overall groundwater metrics.
        """
        overall_groundwater_metrics = OverallGroundwaterMetrics()

        for groundwater_in_doc in self.groundwater_entries:
            filename = groundwater_in_doc.filename
            ground_truth_data = self.ground_truth.for_file(filename)
            pred_to_gt_matching = {v: k for k, v in self.gt_to_pred_matching[filename].items()}

            # lists to contain the metrics
            groundwater_metrics_list = []
            groundwater_depth_metrics_list = []
            groundwater_elevation_metrics_list = []
            groundwater_date_metrics_list = []

            # iterate on all the borehole detected in the document
            for pred_index, groundwaters_in_borehole in enumerate(groundwater_in_doc.borehole_groundwaters):
                # from the matching previously done on the layer description, extract the coresponding gt borehole
                ground_truth_index = pred_to_gt_matching[pred_index]
                gt_borehole_groundwater = ground_truth_data[ground_truth_index]
                if gt_borehole_groundwater is None or gt_borehole_groundwater.get("groundwater") is None:
                    ground_truth = []  # If no ground truth is available, set it to an empty list
                else:
                    ground_truth = gt_borehole_groundwater.get("groundwater")

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

                # TODO store the correctness directly on the Groundwater objects, so we can use that in the
                # visualizations (cf. https://github.com/swisstopo/swissgeol-boreholes-dataextraction/issues/124)
                groundwater_metrics = count_against_ground_truth(
                    [
                        (
                            entry.feature.depth,
                            entry.feature.format_date(),
                            entry.feature.elevation,
                        )
                        for entry in groundwaters_in_borehole.groundwater_feature_list
                    ],
                    [(entry.depth, entry.format_date(), entry.elevation) for entry in gt_groundwater],
                )
                groundwater_depth_metrics = count_against_ground_truth(
                    [entry.feature.depth for entry in groundwaters_in_borehole.groundwater_feature_list],
                    [entry.depth for entry in gt_groundwater],
                )
                groundwater_elevation_metrics = count_against_ground_truth(
                    [entry.feature.elevation for entry in groundwaters_in_borehole.groundwater_feature_list],
                    [entry.elevation for entry in gt_groundwater],
                )
                groundwater_date_metrics = count_against_ground_truth(
                    [entry.feature.format_date() for entry in groundwaters_in_borehole.groundwater_feature_list],
                    [entry.format_date() for entry in gt_groundwater],
                )
                groundwater_metrics_list.append(groundwater_metrics)
                groundwater_depth_metrics_list.append(groundwater_depth_metrics)
                groundwater_elevation_metrics_list.append(groundwater_elevation_metrics)
                groundwater_date_metrics_list.append(groundwater_date_metrics)

            # we take the micro-average across boreholes
            file_groundwater_metrics = GroundwaterMetrics(
                groundwater_metrics=Metrics.micro_average(groundwater_metrics_list),
                groundwater_depth_metrics=Metrics.micro_average(groundwater_depth_metrics_list),
                groundwater_elevation_metrics=Metrics.micro_average(groundwater_elevation_metrics_list),
                groundwater_date_metrics=Metrics.micro_average(groundwater_date_metrics_list),
                filename=filename,
            )

            overall_groundwater_metrics.add_groundwater_metrics(file_groundwater_metrics)

        return overall_groundwater_metrics
