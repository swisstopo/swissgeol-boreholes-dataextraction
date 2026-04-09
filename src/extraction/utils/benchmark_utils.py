"""Utility functions for benchmarking in the extraction pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.mlflow_tracking import mlflow
from extraction.core.extract import ExtractionResult, open_pdf
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


def on_file_done(
    result: ExtractionResult,
    out_directory: Path,
    input_directory: Path,
    write_csv: bool,
    input_is_file: bool,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
) -> None:
    """Write CSV and/or draw visualizations for a single extracted file.

    Intended to be used via functools.partial with all args pre-bound except result.
    MLflow artifact logging for CSV files is performed here when tracking is active.

    Args:
        result (ExtractionResult): Output of extract() for this file.
        out_directory (Path): Directory for output artifacts.
        input_directory (Path): Directory (or file path) of input PDFs.
        write_csv (bool): Whether to write CSV output.
        input_is_file (bool): True if input_directory points to a single file.
        skip_draw_predictions (bool): Skip drawing final prediction overlays. Defaults to False.
        draw_lines (bool): Draw detected lines. Defaults to False.
        draw_tables (bool): Draw detected table structures. Defaults to False.
        draw_strip_logs (bool): Draw detected strip log structures. Defaults to False.
    """
    if write_csv:
        csv_paths = write_csv_for_file(result.predictions, out_directory)
        if mlflow:
            for csv_path in csv_paths:
                mlflow.log_artifact(str(csv_path), "csv")

    if not skip_draw_predictions or draw_lines or draw_tables or draw_strip_logs:
        pdf_path = input_directory if input_is_file else input_directory / result.predictions.file_name
        file_name = result.predictions.file_name
        from extraction.annotations.draw import plot_prediction, plot_strip_logs, plot_tables
        from extraction.annotations.plot_utils import plot_lines, save_visualization

        draw_directory = out_directory / "draw"
        draw_directory.mkdir(parents=True, exist_ok=True)

        with open_pdf(file=pdf_path, filename=file_name) as doc:
            for page_data in result.pages_data:
                page = doc[page_data.page_index]
                page_number = page_data.page_index + 1

                if draw_tables:
                    img = plot_tables(page, page_data.table_structures, page_data.page_index)
                    save_visualization(img, file_name, page_number, "tables", draw_directory)

                if draw_strip_logs:
                    img = plot_strip_logs(page, page_data.strip_logs, page_data.page_index)
                    save_visualization(img, file_name, page_number, "strip_logs", draw_directory)

                if draw_lines:
                    img = plot_lines(page, page_data.lines, scale_factor=line_detection_params["pdf_scale_factor"])
                    save_visualization(img, file_name, page_number, "lines", draw_directory)

            if not skip_draw_predictions:
                plot_prediction(result.predictions, doc, draw_directory)
