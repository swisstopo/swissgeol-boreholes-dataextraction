"""Utility functions for benchmarking in the extraction pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from extraction.core.extract import ExtractionResult, open_pdf
from extraction.features.predictions.file_predictions import FilePredictions
from swissgeol_doc_processing.utils.file_utils import read_params

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

line_detection_params = read_params("line_detection_params.yml")
logger = logging.getLogger(__name__)


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

    def on_file_done(self, result: ExtractionResult, out_directory: Path, pdf_path: Path) -> None:
        """Write CSV and/or draw visualizations for a single extracted file.

        Args:
            result (ExtractionResult): Output of extract() for this file.
            out_directory (Path): Directory for output artifacts.
            pdf_path (Path): File path of the input PDF.
        """
        if self.write_csv:
            write_csv_for_file(result.predictions, out_directory)

        if not self.skip_draw_predictions or self.draw_lines or self.draw_tables or self.draw_strip_logs:
            file_name = result.predictions.file_name
            from extraction.annotations.draw import plot_prediction, plot_strip_logs, plot_tables
            from extraction.annotations.plot_utils import plot_lines, save_visualization

            draw_directory = out_directory / "draw"
            draw_directory.mkdir(parents=True, exist_ok=True)

            with open_pdf(file=pdf_path, filename=file_name) as doc:
                for page_data in result.pages_data:
                    page = doc[page_data.page_index]
                    page_number = page_data.page_index + 1

                    if self.draw_tables:
                        img = plot_tables(page, page_data.table_structures, page_data.page_index)
                        save_visualization(img, file_name, page_number, "tables", draw_directory)

                    if self.draw_strip_logs:
                        img = plot_strip_logs(page, page_data.strip_logs, page_data.page_index)
                        save_visualization(img, file_name, page_number, "strip_logs", draw_directory)

                    if self.draw_lines:
                        img = plot_lines(page, page_data.lines, scale_factor=line_detection_params["pdf_scale_factor"])
                        save_visualization(img, file_name, page_number, "lines", draw_directory)

                if not self.skip_draw_predictions:
                    plot_prediction(result.predictions, doc, draw_directory)
