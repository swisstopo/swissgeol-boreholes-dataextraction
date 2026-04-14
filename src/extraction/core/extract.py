"""Core extraction pipeline for borehole data from PDF documents."""

import dataclasses
import logging
from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pymupdf

from extraction.features.extract import extract_page
from extraction.features.groundwater.groundwater_extraction import (
    GroundwaterInDocument,
    GroundwaterLevelExtractor,
)
from extraction.features.metadata.borehole_name_extraction import NameInDocument, extract_borehole_names
from extraction.features.metadata.metadata import FileMetadata, MetadataInDocument
from extraction.features.predictions.borehole_predictions import BoreholePredictions
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.predictions.predictions import BoreholeListBuilder
from extraction.features.stratigraphy.layer.continuation_detection import merge_boreholes
from extraction.features.stratigraphy.layer.layer import LayersInDocument
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.geometry.line_detection import extract_lines
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics
from swissgeol_doc_processing.utils.file_utils import read_params
from swissgeol_doc_processing.utils.strip_log_detection import StripLog, detect_strip_logs
from swissgeol_doc_processing.utils.table_detection import TableStructure, detect_table_structures

matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")
name_detection_params = read_params("name_detection_params.yml")
table_detection_params = read_params("table_detection_params.yml")
striplog_detection_params = read_params("striplog_detection_params.yml")

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PageExtractionData:
    """Intermediate per-page detection data, needed for optional visualization."""

    page_index: int
    lines: list[Line]
    table_structures: list[TableStructure]
    strip_logs: list[StripLog]


@dataclasses.dataclass
class ExtractionResult:
    """Full output of extract(): predictions and per-page data needed for visualization."""

    predictions: FilePredictions
    pages_data: list[PageExtractionData]


@contextmanager
def open_pdf(
    file: Path | BytesIO,
    filename: str = None,
) -> Generator[pymupdf.Document, None, None]:
    """Open a PDF document from either a file path or a binary stream.

    This context manager handles opening and closing a PyMuPDF document,
    accepting either a file path or an already-open binary stream.

    Args:
        file (Path | BytesIO): Either a Path to a PDF file or an open binary stream (BytesIO).
        filename (str): Filename to associate with the document. Only used when 'file' is a stream.

    Yields:
        pymupdf.Document: The opened PDF document.
    """
    doc = (
        pymupdf.Document(filename=filename, stream=file)
        if isinstance(file, BytesIO)
        else pymupdf.Document(filename=file)
    )
    yield doc
    doc.close()


def extract(
    file: Path | BytesIO,
    filename: str,
    part: str = "all",
    analytics: MatchingParamsAnalytics | None = None,
) -> ExtractionResult:
    """Extract pipeline for input file.

    Core extraction only. Pass an on_file_done callback to run_extraction_predictions() for visualizations and CSV
    output.

    Args:
        file (Path | BytesIO): Path or stream of file to process.
        filename (str): Name of the file used as identifier.
        part (str): Pipeline mode, "all" for full extraction, "metadata" for metadata only. Defaults to "all".
        analytics (MatchingParamsAnalytics): Analytics object for tracking matching parameters. Defaults to None.

    Returns:
        ExtractionResult: Predictions and per-page detection data for the input file.
    """
    # Clear cache to avoid cache contamination across different files, which can cause incorrect
    # visualizations; see also https://github.com/swisstopo/swissgeol-boreholes-suite/issues/1935
    pymupdf.TOOLS.store_shrink(100)

    with open_pdf(file=file, filename=filename) as doc:
        # Extract metadata
        file_metadata = FileMetadata.from_document(doc, matching_params)
        metadata = MetadataInDocument.from_document(doc, file_metadata.language, matching_params)

        # Save the predictions to the overall predictions object, initialize common variables
        all_groundwater_entries = GroundwaterInDocument([], filename)
        all_name_entries = NameInDocument([], filename)
        boreholes_per_page = []
        pages_data = []

        if part != "all":
            return ExtractionResult(predictions=FilePredictions([], file_metadata, filename), pages_data=[])

        # Extract the Flayers
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            logger.info(f"Processing page {page_number}")

            text_lines = extract_text_lines(page)
            long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)
            name_entries = extract_borehole_names(text_lines, name_detection_params)
            all_name_entries.name_feature_list.extend(name_entries)

            # Detect table structures on the page
            table_structures = detect_table_structures(
                page, long_or_horizontal_lines, text_lines, table_detection_params
            )

            # Detect strip logs on the page
            strip_logs = detect_strip_logs(page, text_lines, striplog_detection_params)

            # Extract the stratigraphy
            page_layers = extract_page(
                text_lines,
                long_or_horizontal_lines,
                all_geometric_lines,
                table_structures,
                strip_logs,
                file_metadata.language,
                page_index,
                page,
                line_detection_params,
                analytics,
                **matching_params,
            )
            boreholes_per_page.append(page_layers)

            # Extract the groundwater levels
            groundwater_extractor = GroundwaterLevelExtractor(file_metadata.language, matching_params)
            groundwater_entries = groundwater_extractor.extract_groundwater(
                page_number=page_number,
                text_lines=text_lines,
                geometric_lines=long_or_horizontal_lines,
                extracted_boreholes=page_layers,
            )
            all_groundwater_entries.groundwater_feature_list.extend(groundwater_entries)

            # Store per-page intermediate data for optional downstream visualization
            pages_data.append(
                PageExtractionData(
                    page_index=page_index,
                    lines=all_geometric_lines,
                    table_structures=table_structures,
                    strip_logs=strip_logs,
                )
            )

        # Merge detections if possible
        layers_with_bb_in_document = LayersInDocument(merge_boreholes(boreholes_per_page, matching_params), filename)

        # create list of BoreholePrediction objects with all the separate lists
        borehole_predictions_list: list[BoreholePredictions] = BoreholeListBuilder(
            layers_with_bb_in_document=layers_with_bb_in_document,
            file_name=filename,
            groundwater_in_doc=all_groundwater_entries,
            names_in_doc=all_name_entries,
            elevations_list=metadata.elevations,
            coordinates_list=metadata.coordinates,
        ).build()

        # now that the matching is done, duplicated groundwater can be removed and depths info can be set
        for borehole in borehole_predictions_list:
            borehole.filter_groundwater_entries()

        return ExtractionResult(
            predictions=FilePredictions(borehole_predictions_list, file_metadata, filename),
            pages_data=pages_data,
        )
