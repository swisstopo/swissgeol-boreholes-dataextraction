"""Minimal borehole identification pipeline for XGBoost classification.

This module provides a simplified extraction pipeline that generates features
for machine learning classification. Instead of full borehole data extraction,
it extracts key features that correlate with borehole presence.
"""

from dataclasses import dataclass

import pymupdf
from dotenv import load_dotenv

from extraction.features.extract import extract_page, extract_sidebar_information
from extraction.features.metadata.borehole_name_extraction import extract_borehole_names
from extraction.features.stratigraphy.sidebar.classes.sidebar import SidebarQualityMetrics
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.geometry.line_detection import extract_lines
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.text.stemmer import find_matching_expressions
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.file_utils import read_params
from swissgeol_doc_processing.utils.strip_log_detection import StripLog, detect_strip_logs
from swissgeol_doc_processing.utils.table_detection import TableStructure, detect_table_structures

load_dotenv()


def load_default_params() -> dict:
    """Load all default extraction parameters.

    Returns:
        dict with keys: matching_params, line_detection_params,
        name_detection_params, table_detection_params, striplog_detection_params
    """
    return {
        "matching_params": read_params("matching_params.yml"),
        "line_detection_params": read_params("line_detection_params.yml"),
        "name_detection_params": read_params("name_detection_params.yml"),
        "table_detection_params": read_params("table_detection_params.yml"),
        "striplog_detection_params": read_params("striplog_detection_params.yml"),
    }


@dataclass
class ExtractionContext:
    """Context object containing pre-extracted page data to avoid re-extraction.

    This pattern allows passing cached extraction results to feature extraction
    functions.

    Usage:
        params = load_default_params()

        # Extract and cache all page data
        context = ExtractionContext.from_page(
            page,
            params["line_detection_params"],
            params["striplog_detection_params"],
            params["table_detection_params"],
        )

        # Reuse in feature extraction
        features = extract_page_features(
            page, page_index, language, extraction_context=context, **params
        )
    """

    text_lines: list[TextLine]
    long_or_horizontal_lines: list[Line]
    all_geometric_lines: list[Line]
    strip_logs: list[StripLog]
    table_structures: list[TableStructure]
    language: str | None = None

    @classmethod
    def from_page(
        cls,
        page: pymupdf.Page,
        line_detection_params: dict,
        striplog_detection_params: dict,
        table_detection_params: dict,
    ) -> "ExtractionContext":
        """Factory method to extract and cache all data from a page.

        Args:
            page: The PDF page to extract from
            line_detection_params: Parameters for line detection
            striplog_detection_params: Parameters for strip log detection
            table_detection_params: Parameters for table detection

        Returns:
            ExtractionContext with all data extracted
        """
        text_lines = extract_text_lines(page)
        long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)
        strip_logs = detect_strip_logs(page, text_lines, striplog_detection_params)
        table_structures = detect_table_structures(page, long_or_horizontal_lines, text_lines, table_detection_params)

        return cls(
            text_lines=text_lines,
            long_or_horizontal_lines=long_or_horizontal_lines,
            all_geometric_lines=all_geometric_lines,
            strip_logs=strip_logs,
            table_structures=table_structures,
        )


@dataclass
class ExtractedPageFeatures:
    """Features extracted from a single page for borehole identification.

    Attributes:
        page_number: Page index.
        number_of_valid_borehole_descriptions: Count of valid material descriptions.
        number_of_strip_logs: Count of detected strip logs.
        number_of_tables: Count of detected tables.
        number_of_boreholes: Count of detected boreholes.
        sidebar_information: Sidebar extraction quality metrics.
        number_long_or_horizontal_lines: Count of long/horizontal lines.
        number_all_geometric_lines: Count of all geometric lines.
        text_line_count: Number of text lines on page.
        borehole_name_entries: List of detected borehole names with confidence scores.
    """

    page_number: int
    number_of_valid_borehole_descriptions: int
    number_of_strip_logs: int
    number_of_tables: int
    number_of_boreholes: int
    sidebar_information: SidebarQualityMetrics
    number_long_or_horizontal_lines: int
    number_all_geometric_lines: int
    text_line_count: int
    borehole_name_entries: list[dict]


def extract_page_features(
    page: pymupdf.Page,
    page_index: int,
    language: str,
    matching_params: dict,
    line_detection_params: dict,
    name_detection_params: dict,
    table_detection_params: dict,
    striplog_detection_params: dict,
    extraction_context: ExtractionContext | None = None,
    extract_boreholes: bool = False,
) -> ExtractedPageFeatures:
    """Extract features from a single page for borehole identification.

    Args:
        page (pymupdf.Page): The PDF page to process.
        page_index (int): The index of the page.
        language (str): The detected language of the document.
        matching_params (dict): Parameters for material description matching.
        line_detection_params (dict): Parameters for line detection.
        name_detection_params (dict): Parameters for borehole name detection.
        table_detection_params (dict): Parameters for table detection.
        striplog_detection_params (dict): Parameters for strip log detection.
        extraction_context (Optional[ExtractionContext]): Pre-extracted page data to avoid re-extraction.
        extract_boreholes (bool): Whether to extract actual borehole data or just features.

    Returns:
        ExtractedPageFeatures: Features extracted from the page for borehole identification.
    """
    if extraction_context is not None:
        text_lines = extraction_context.text_lines
        long_or_horizontal_lines = extraction_context.long_or_horizontal_lines
        all_geometric_lines = extraction_context.all_geometric_lines
        strip_logs = extraction_context.strip_logs
        table_structures = extraction_context.table_structures
    else:
        text_lines = extract_text_lines(page)
        long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)
        strip_logs = detect_strip_logs(page, text_lines, striplog_detection_params)
        table_structures = detect_table_structures(page, long_or_horizontal_lines, text_lines, table_detection_params)

    number_of_strip_logs = len(strip_logs)
    number_of_tables = len(table_structures)

    # Extract material descriptions by counting lines with geological material keywords
    # Get the material keywords (including_expressions) for the detected language
    material_description_config = matching_params.get("material_description", {})
    language_config = material_description_config.get(language, {})
    material_keywords = language_config.get("including_expressions", [])
    split_threshold = matching_params.get("compound_split_threshold", 0.4)

    valid_descriptions = []
    if material_keywords:
        for text_line in text_lines:
            # Check if line contains material keywords
            # find_matching_expressions signature: (patterns, split_threshold, targets, language, ...)
            # patterns = material keywords to find
            # targets = text to search in
            if find_matching_expressions(material_keywords, split_threshold, [text_line.text], language):
                valid_descriptions.append(text_line)

    number_of_valid_borehole_descriptions = len(valid_descriptions)

    sidebar_information = extract_sidebar_information(
        text_lines,
        long_or_horizontal_lines,
        all_geometric_lines,
        table_structures,
        strip_logs,
        language,
        page_index,
        page,
        line_detection_params,
        None,  # analytics parameter
        **matching_params,
    )

    if extract_boreholes:
        borehole_count = extract_page(
            text_lines,
            long_or_horizontal_lines,
            all_geometric_lines,
            table_structures,
            strip_logs,
            language,
            page_index,
            page,
            line_detection_params,
            None,  # analytics parameter
            **matching_params,
        )
    else:
        borehole_count = 0

    number_of_boreholes = len(borehole_count)

    # Extract borehole names
    name_entries = extract_borehole_names(text_lines, name_detection_params)
    borehole_name_entries = [
        {
            "name": entry.feature.name,
            "confidence": entry.feature.confidence,
        }
        for entry in name_entries
    ]

    return ExtractedPageFeatures(
        page_number=page_index,
        number_of_valid_borehole_descriptions=number_of_valid_borehole_descriptions,
        number_of_strip_logs=number_of_strip_logs,
        number_of_tables=number_of_tables,
        number_of_boreholes=number_of_boreholes,
        sidebar_information=sidebar_information,
        number_long_or_horizontal_lines=len(long_or_horizontal_lines),
        number_all_geometric_lines=len(all_geometric_lines),
        text_line_count=len(text_lines),
        borehole_name_entries=borehole_name_entries,
    )
