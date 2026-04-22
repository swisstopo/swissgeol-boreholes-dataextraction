"""Classes for keeping track of metrics such as the F1-score, precision and recall."""

from collections import defaultdict
from collections.abc import Callable
from functools import reduce

import pandas as pd

from core.benchmark_utils import Metrics


class OverallMetrics:
    """Keeps track of a particular metrics for all documents in a dataset."""

    # TODO: Currently, some methods for averaging metrics are in the Metrics class.
    # (see micro_average(metric_list: list["Metrics"]). On the long run, we should refactor
    # this to have a single place where these averaging computations are implemented.

    def __init__(self, metrics: dict[str, Metrics] | None = None):
        """Initialise class with an optional pre-populated metrics dictionary.

        Args:
            metrics (dict[str, Metrics] | None): Mapping of document filename to per-document
                Metrics. Defaults to an empty dict when None.
        """
        self.metrics: dict[str, Metrics] = metrics if metrics is not None else {}

    def macro_f1(self) -> float:
        """Compute the macro F1 score."""
        if self.metrics:
            return sum([metric.f1 for metric in self.metrics.values()]) / len(self.metrics)
        else:
            return 0

    def macro_precision(self) -> float:
        """Compute the macro precision score."""
        if self.metrics:
            return sum([metric.precision for metric in self.metrics.values()]) / len(self.metrics)
        else:
            return 0

    def macro_recall(self) -> float:
        """Compute the macro recall score."""
        if self.metrics:
            return sum([metric.recall for metric in self.metrics.values()]) / len(self.metrics)
        else:
            return 0

    def get_language_subset(self, fp_languages: dict[str, str], language: str) -> "OverallMetrics":
        """Filter per-file metrics to only those whose file language matches the given language code.

        Args:
            fp_languages (dict[str, str]): Dictionary with filenames as key and language as value.
            language (str): Language code to filter by (e.g. "de", "fr").

        Returns:
            OverallMetrics: Metrics for files matching the specified language.
        """
        return OverallMetrics(
            {filename: metric for filename, metric in self.metrics.items() if fp_languages.get(filename) == language}
        )

    def to_macro_dict(self, prefix: str) -> dict[str, float]:
        """Return macro-averaged f1, recall, and precision as a dictionary with prefixed keys.

        Args:
            prefix (str): String prepended to each key.

        Returns:
            dict[str, float]: Flat macro-averaged metrics dictionary with prefixed keys.
        """
        return {
            f"{prefix}_f1": self.macro_f1(),
            f"{prefix}_recall": self.macro_recall(),
            f"{prefix}_precision": self.macro_precision(),
        }

    def to_micro_dict(self, prefix: str) -> dict[str, float]:
        """Return micro-averaged f1, recall, and precision as a dictionary with prefixed keys.

        Args:
            prefix (str): String prepended to each key.

        Returns:
            dict[str, float]: Flat micro-averaged metrics dictionary with prefixed keys.
        """
        micro_avg = Metrics.micro_average(self.metrics.values())
        return {
            f"{prefix}_f1": micro_avg.f1,
            f"{prefix}_recall": micro_avg.recall,
            f"{prefix}_precision": micro_avg.precision,
        }

    def to_df(
        self,
        prefix: str,
        extra_metrics: dict[str, Callable[[Metrics], float]] | None = None,
    ) -> pd.DataFrame:
        """Build a per-file DataFrame with f1, recall, and precision columns.

        Args:
            prefix (str): String prepended to each metric column name.
            extra_metrics (dict[str, Callable[[Metrics], float]] | None): Dictionary of optional
                metrics as a callable function. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame indexed by filename with one prefixed column per metric.
        """
        metrics = {"f1": lambda m: m.f1, "recall": lambda m: m.recall, "precision": lambda m: m.precision}
        if extra_metrics:
            metrics = metrics | extra_metrics

        return pd.DataFrame(
            {
                "filename": filename,
                **{f"{prefix}_{name}": fn(metric) for name, fn in metrics.items()},
            }
            for filename, metric in self.metrics.items()
        ).set_index("filename")


class OverallMetricsCatalog:
    """Keeps track of all different relevant metrics that are computed for a dataset."""

    def __init__(self, languages: set[str]):
        self.layer_metrics = OverallMetrics()
        self.material_description_metrics = OverallMetrics()
        self.depth_interval_metrics = OverallMetrics()
        self.groundwater_metrics = OverallMetrics()
        self.groundwater_depth_metrics = OverallMetrics()
        self.elevation_metrics = OverallMetrics()
        self.coordinates_metrics = OverallMetrics()
        self.name_metrics = OverallMetrics()
        self.languages = languages

        # Initialize language-specific metrics
        for lang in languages:
            setattr(self, f"{lang}_layer_metrics", OverallMetrics())
            setattr(self, f"{lang}_depth_interval_metrics", OverallMetrics())
            setattr(self, f"{lang}_material_description_metrics", OverallMetrics())

    def add_datapoint(
        self,
        filename: str,
        material_description_metric: Metrics,
        layer_metrics: Metrics,
        depth_interval_metric: Metrics,
        elevation_metric: Metrics,
        coordinates_metric: Metrics,
        name_metric: Metrics,
    ) -> None:
        """Register per-file metrics for all extraction categories.

        Args:
            filename (str): The file name used as key.
            material_description_metric (Metrics): Material description metrics.
            layer_metrics (Metrics): Layer detection metrics.
            depth_interval_metric (Metrics): Depth interval metrics.
            elevation_metric (Metrics): Elevation metrics.
            coordinates_metric (Metrics): Coordinate metrics.
            name_metric (Metrics): Borehole name metrics.
        """
        self.layer_metrics.metrics[filename] = layer_metrics
        self.material_description_metrics.metrics[filename] = material_description_metric
        self.depth_interval_metrics.metrics[filename] = depth_interval_metric
        self.elevation_metrics.metrics[filename] = elevation_metric
        self.coordinates_metrics.metrics[filename] = coordinates_metric
        self.name_metrics.metrics[filename] = name_metric

    def to_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return document-level metrics as two DataFrames, one for metadata and one for geology.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A pair (metadata_df, geology_df), both indexed
                by filename.
        """
        return self.to_metadata_df(), self.to_geology_df()

    def to_metadata_df(self) -> pd.DataFrame:
        """Build a per-file DataFrame with name, coordinate, and elevation metrics.

        Returns:
            pd.DataFrame: DataFrame indexed by filename with columns for name, coordinate, and
                elevation f1/recall/precision.
        """
        dfs = [
            self.name_metrics.to_df("name"),
            self.coordinates_metrics.to_df("coordinate"),
            self.elevation_metrics.to_df("elevation"),
        ]

        df_final = reduce(lambda left, right: left.merge(right, on="filename"), dfs)
        return df_final.sort_index(ascending=True)

    def to_geology_df(self) -> pd.DataFrame:
        """Build a per-file DataFrame with layer, depth interval, material description, and groundwater metrics.

        Layer columns also include layer_num_total: (tp + fn) and layer_num_wrong: (fp + fn).
        Groundwater depth columns also include groundwater_depth_num_detected: (tp + fp).

        Returns:
            pd.DataFrame: DataFrame indexed by filename with all geology metric columns,
                sorted alphabetically by filename.
        """
        dfs = [
            self.layer_metrics.to_df(
                "layer",
                extra_metrics={
                    "num_total": lambda metric: metric.tp + metric.fn,
                    "num_wrong": lambda metric: metric.fp + metric.fn,
                },
            ),
            self.material_description_metrics.to_df("material_description"),
            self.depth_interval_metrics.to_df("depth_interval"),
            self.groundwater_metrics.to_df("groundwater"),
            self.groundwater_depth_metrics.to_df(
                "groundwater_depth",
                extra_metrics={"num_detected": lambda metric: metric.tp + metric.fp},
            ),
        ]

        df_final = reduce(lambda left, right: left.merge(right, on="filename"), dfs)
        return df_final.sort_index(ascending=True)

    def to_dicts(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return overall metrics as two flat dictionaries, one for metadata and one for geology.

        Returns:
            tuple[dict[str, float], dict[str, float]]: A pair (metadata_dict, geology_dict).
        """
        return self.to_metadata_dict(), self.to_geology_dict()

    def to_geology_dict(self) -> dict[str, float]:
        """Return overall geology metrics as a dictionary.

        Includes macro-averaged layer (and language-specific), depth interval, and material
        description metrics, annd micro-averaged groundwater metrics.

        Returns:
            dict[str, float]: Dictionary of overall geology metrics with prefixed keys.
        """
        # Initialize a defaultdict to automatically return 0.0 for missing keys
        result = defaultdict(lambda: None)

        # Compute the micro-average metrics for the groundwater and groundwater depth metrics
        groundwater_metrics = Metrics.micro_average(self.groundwater_metrics.metrics.values())
        groundwater_depth_metrics = Metrics.micro_average(self.groundwater_depth_metrics.metrics.values())

        # Populate the basic metrics
        result.update(self.layer_metrics.to_macro_dict("layer"))
        result.update(self.depth_interval_metrics.to_macro_dict("depth_interval"))
        result.update(self.material_description_metrics.to_macro_dict("material_description"))
        result.update(groundwater_metrics.to_dict("groundwater"))
        result.update(groundwater_depth_metrics.to_dict("groundwater_depth"))

        # Add dynamic language-specific metrics only if they exist
        for lang in self.languages:
            for metric_topic in ["layer", "depth_interval", "material_description"]:
                key_prefix = f"{lang}_{metric_topic}"
                key = f"{key_prefix}_metrics"

                if getattr(self, key) and getattr(self, key).metrics:
                    result.update(getattr(self, key).to_macro_dict(key_prefix))

        return dict(result)  # Convert defaultdict back to a regular dict

    def to_metadata_dict(self) -> dict[str, float]:
        """Return overall metadata metrics as a dictionary.

        Computes micro-averages across all files for name, elevation, and coordinate metrics.

        Returns:
            dict[str, float]:  Dictionary with prefixed keys.
        """
        return (
            self.name_metrics.to_micro_dict("name")
            | self.elevation_metrics.to_micro_dict("elevation")
            | self.coordinates_metrics.to_micro_dict("coordinates")
        )
