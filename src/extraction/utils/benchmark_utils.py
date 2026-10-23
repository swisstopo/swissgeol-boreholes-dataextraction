"""Utility functions for benchmarking in the extraction pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from core.mlflow_tracking import mlflow
from extraction.annotations.draw import plot_prediction, plot_strip_logs, plot_tables
from extraction.annotations.plot_utils import plot_lines, save_visualization
from extraction.core.extract import ExtractionResult, PageExtractionData, open_pdf
from extraction.evaluation.benchmark.score import ExtractionBenchmarkSummary
from extraction.features.predictions.file_predictions import FilePredictions
from swissgeol_doc_processing.utils.file_utils import read_params

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

line_detection_params = read_params("line_detection_params.yml")
logger = logging.getLogger(__name__)


def log_metric_mlflow(
    summary: ExtractionBenchmarkSummary,
    out_dir: Path,
    artifact_name: str = "benchmark_summary.json",
) -> None:
    """Log benchmark metrics + a JSON summary artifact to the *currently active* MLflow run.

    - Does NOT start/end MLflow runs.
    """
    metrics = summary.metrics_flat(short=True)
    mlflow.log_metrics(metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / artifact_name
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(summary.model_dump(), f, ensure_ascii=False, indent=2)

    mlflow.log_artifact(str(summary_path), artifact_path="summary")


def write_csv_for_file(predictions: FilePredictions, out_directory: Path) -> list[Path]:
    """Write per-borehole CSV files for a single file's predictions.

    Args:
        predictions (FilePredictions): Predictions for a single file.
        out_directory (Path): Directory under which a "csv/" sub-folder is created.

    Returns:
        list[Path]: Paths of the written CSV files.
    """
    csv_directory = out_directory / "csv"
    csv_directory.mkdir(parents=True, exist_ok=True)
    base_path = csv_directory / Path(predictions.file_name).stem
    csv_paths = []
    for index, borehole in enumerate(predictions.borehole_predictions_list):
        csv_path = (
            Path(f"{base_path}_{index}.csv")
            if len(predictions.borehole_predictions_list) > 1
            else Path(f"{base_path}.csv")
        )
        logger.info(f"Writing CSV predictions to {csv_path}")
        with open(csv_path, "w", encoding="utf8", newline="") as csvfile:
            csvfile.write(borehole.to_csv())
        csv_paths.append(csv_path)
    return csv_paths


@dataclass
class CallbackFactory:
    """Contains callback methods for the data extraction pipeline."""

    write_csv: bool
    skip_draw_predictions: bool
    draw_lines: bool
    draw_tables: bool
    draw_strip_logs: bool
    out_directory: Path

    def on_page_done(self, page: pymupdf.Page, page_data: PageExtractionData) -> None:
        scale_factor = page_data.line_detection_result.cv_scale_factor
        page_number = page_data.page_index + 1

        file_name = page_data.file_name

        draw_directory = self.out_directory / "draw"
        draw_directory.mkdir(parents=True, exist_ok=True)

        if self.draw_tables:
            img = plot_tables(page, page_data.table_structures, page_data.page_index, scale_factor)
            save_visualization(img, file_name, page_number, "tables", draw_directory)

        if self.draw_strip_logs:
            img = plot_strip_logs(page, page_data.strip_logs, page_data.page_index, scale_factor)
            save_visualization(img, file_name, page_number, "strip_logs", draw_directory)

        if self.draw_lines:
            img = plot_lines(page, page_data.line_detection_result)
            save_visualization(img, file_name, page_number, "lines", draw_directory)

    def on_file_done(self, result: ExtractionResult, pdf_path: Path) -> None:
        """Write CSV and/or draw visualizations for a single extracted file.

        MLflow artifact logging for CSV files is performed here when tracking is active.

        Args:
            result (ExtractionResult): Output of extract() for this file.
            pdf_path (Path): File path of the input PDF.
        """
        if self.write_csv:
            csv_paths = write_csv_for_file(result.predictions, self.out_directory)
            if mlflow:
                for csv_path in csv_paths:
                    mlflow.log_artifact(str(csv_path), "csv")

        if not self.skip_draw_predictions:
            file_name = result.predictions.file_name

            draw_directory = self.out_directory / "draw"
            draw_directory.mkdir(parents=True, exist_ok=True)

            logger.info("Drawing predictions for file %s", file_name)
            with open_pdf(file=pdf_path, filename=file_name) as doc:
                for page_index, page in enumerate(doc):
                    scale_factor = result.cv_scale_factor[page_index]
                    plot_prediction(result.predictions, page, draw_directory, scale_factor)

            logger.info("Finished drawing predictions for file %s", file_name)
