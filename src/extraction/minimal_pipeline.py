"""Minimal borehole identification pipeline for XGBoost classification.

This module provides a simplified extraction pipeline that generates features
for machine learning classification. Instead of full borehole data extraction,
it extracts key features that correlate with borehole presence.

Example usage:
    import pymupdf
    from extraction.minimal_pipeline import extract_borehole_features

    with pymupdf.Document("path/to/borehole.pdf") as doc:
        features = extract_borehole_features(doc)
"""
import logging

import pymupdf
from dotenv import load_dotenv

from extraction.features.extract import extract_page, extract_sidebar_information
from extraction.features.metadata.borehole_name_extraction import extract_borehole_names
from extraction.features.metadata.metadata import FileMetadata, MetadataInDocument

from swissgeol_doc_processing.geometry.line_detection import extract_lines
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.text.stemmer import find_matching_expressions
from swissgeol_doc_processing.utils.file_utils import read_params
from swissgeol_doc_processing.utils.language_detection import detect_language_of_document
from swissgeol_doc_processing.utils.strip_log_detection import StripLog, detect_strip_logs
from swissgeol_doc_processing.utils.table_detection import detect_table_structures, TableStructure
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line

from dataclasses import dataclass
from typing import Optional
import time
from concurrent.futures import as_completed, ThreadPoolExecutor

load_dotenv()

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Load configuration files
matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")
name_detection_params = read_params("name_detection_params.yml")
striplog_detection_params = read_params("striplog_detection_params.yml")
table_detection_params = read_params("table_detection_params.yml")

@dataclass
class ExtractionContext:
    """Context object containing pre-extracted page data to avoid re-extraction.

    This pattern allows passing cached extraction results to feature extraction
    functions, improving performance by ~40-50% when data is already available.

    Usage:
        # Extract once
        context = ExtractionContext()
        context.text_lines = extract_text_lines(page)
        context.long_or_horizontal_lines, context.all_geometric_lines = extract_lines(page, params)

        # Reuse in feature extraction
        features = extract_page_features(page, page_index, language, ..., extraction_context=context)
    """
    text_lines: Optional[list[TextLine]] = None
    long_or_horizontal_lines: Optional[list[Line]] = None
    all_geometric_lines: Optional[list[Line]] = None
    strip_logs: Optional[list[StripLog]] = None
    table_structures: Optional[list[TableStructure]] = None
    language: Optional[str] = None

    @classmethod
    def from_page(cls, page: pymupdf.Page, line_detection_params: dict) -> "ExtractionContext":
        """Factory method to extract and cache all data from a page.

        Args:
            page: The PDF page to extract from
            line_detection_params: Parameters for line detection

        Returns:
            ExtractionContext with all data extracted
        """
        text_lines = extract_text_lines(page)
        long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)

        return cls(
            text_lines=text_lines,
            long_or_horizontal_lines=long_or_horizontal_lines,
            all_geometric_lines=all_geometric_lines,
        )


def extract_page_features(
    page: pymupdf.Page,
    page_index: int,
    language: str,
    matching_params: dict,
    line_detection_params: dict,
    name_detection_params: dict,
    extraction_context: Optional[ExtractionContext] = None,
    extract_boreholes: bool = False,
) -> dict:
    """Extract features from a single page for borehole identification.

    Args:
        page (pymupdf.Page): The PDF page to process.
        page_index (int): The index of the page.
        language (str): The detected language of the document.
        matching_params (dict): Parameters for material description matching.
        line_detection_params (dict): Parameters for line detection.
        name_detection_params (dict): Parameters for borehole name detection.
        extraction_context (Optional[ExtractionContext]): Pre-extracted page data to avoid re-extraction.
        extract_boreholes (bool): Whether to extract actual borehole data or just features.

    Returns:
        dict: Dictionary containing extracted features:
            - page_number: Page index
            - number_of_valid_borehole_descriptions: Count of valid material descriptions
            - number_of_strip_logs: Count of detected strip logs
            - number_of_tables: Count of detected tables
            - number_of_boreholes: Count of detected boreholes
            - sidebar_information: Sidebar extraction results
            - number_long_or_horizontal_lines: Count of long/horizontal lines
            - number_all_geometric_lines: Count of all geometric lines
            - text_line_count: Number of text lines on page
            - borehole_name_entries: List of detected borehole names with confidence scores
    """

    if extraction_context is not None and extraction_context.text_lines is not None:
        text_lines = extraction_context.text_lines
    else:
        text_lines = extract_text_lines(page)

    if (extraction_context is not None
        and extraction_context.long_or_horizontal_lines is not None
        and extraction_context.all_geometric_lines is not None):
        long_or_horizontal_lines = extraction_context.long_or_horizontal_lines
        all_geometric_lines = extraction_context.all_geometric_lines
    else:
        long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)

    # If strip_logs already extracted in context, reuse them
    if extraction_context is not None and extraction_context.strip_logs is not None:
        strip_logs = extraction_context.strip_logs
    else:
        striplog_detection_params = read_params("striplog_detection_params.yml")
        strip_logs = detect_strip_logs(page, text_lines, striplog_detection_params)

    number_of_strip_logs = len(strip_logs)

    # If tables already extracted in context, reuse them
    if extraction_context is not None and extraction_context.table_structures is not None:
        table_structures = extraction_context.table_structures
    else:
        table_detection_params = read_params("table_detection_params.yml")
        table_structures = detect_table_structures(page, long_or_horizontal_lines, text_lines, table_detection_params)

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
            None,
            **matching_params,
        )
    else:
        borehole_count = []

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

    return {
        "page_number": page_index,
        "number_of_valid_borehole_descriptions": number_of_valid_borehole_descriptions,
        "number_of_strip_logs": number_of_strip_logs,
        "number_of_tables": number_of_tables,
        "number_of_boreholes": number_of_boreholes,
        "sidebar_information": sidebar_information,
        "number_long_or_horizontal_lines": len(long_or_horizontal_lines),
        "number_all_geometric_lines": len(all_geometric_lines),
        "text_line_count": len(text_lines),
        "borehole_name_entries": borehole_name_entries,
    }


def extract_borehole_features(
    doc: pymupdf.Document,
    extract_boreholes: bool = False,
    use_parallel: bool = False,
    max_workers: Optional[int] = None,
) -> dict:
    """Extract borehole features from an entire PDF document.

    Args:
        doc (pymupdf.Document): The PDF document to process.
        extract_boreholes (bool): Whether to extract actual borehole data or just features.
        use_parallel (bool): Whether to use parallel processing (ThreadPoolExecutor).
        max_workers (Optional[int]): Max parallel workers (None = CPU count). Only used if use_parallel=True.

    Returns:
        dict: Dictionary containing:
            - language: Detected document language
            - page_features: List of feature dictionaries per page
            - file_features: Aggregated file-level statistics
            - execution_time_seconds: Total processing time
    """
    start_time = time.time()

    # Detect document language
    language = detect_language_of_document(
        doc, matching_params["default_language"], list(matching_params["material_description"].keys())
    )

    logger.info(f"Processing {len(doc)} pages in {'parallel' if use_parallel else 'sequential'} mode")

    page_features = []

    if use_parallel:
        # Use ThreadPoolExecutor for parallel processing (works with pymupdf.Page objects)
        def extract_single_page(page_idx):
            page = doc[page_idx]
            return extract_page_features(
                page, page_idx, language,
                matching_params, line_detection_params, name_detection_params,
                extract_boreholes=extract_boreholes
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_single_page, i): i for i in range(len(doc))}
            results = []

            for future in as_completed(futures):
                page_idx = futures[future]
                results.append((page_idx, future.result()))

            # Sort by page index to maintain order
            results.sort(key=lambda x: x[0])
            page_features = [r[1] for r in results]
    else:
        # Sequential processing
        for page_index, page in enumerate(doc):
            features = extract_page_features(
                page, page_index, language,
                matching_params, line_detection_params, name_detection_params,
                extract_boreholes=extract_boreholes
            )
            page_features.append(features)

    # Calculate file-level aggregated statistics
    total_boreholes = sum(f["number_of_boreholes"] for f in page_features)
    total_descriptions = sum(f["number_of_valid_borehole_descriptions"] for f in page_features)
    total_strip_logs = sum(f["number_of_strip_logs"] for f in page_features)
    total_tables = sum(f["number_of_tables"] for f in page_features)
    pages_with_boreholes = sum(1 for f in page_features if f["number_of_boreholes"] > 0)

    # Calculate confidence metrics
    pages_with_names = sum(1 for f in page_features if len(f["borehole_name_entries"]) > 0)
    pages_with_sidebar = sum(1 for f in page_features if f["sidebar_information"] is not None)

    # Simple confidence heuristic: pages with multiple indicators
    max_confidence = 0.0
    for features in page_features:
        page_confidence = 0.0
        if features["number_of_boreholes"] > 0:
            page_confidence += 0.3
        if len(features["borehole_name_entries"]) > 0:
            page_confidence += 0.2
        if features["sidebar_information"] is not None:
            page_confidence += 0.2
        if features["number_of_valid_borehole_descriptions"] > 0:
            page_confidence += 0.2
        if features["number_of_strip_logs"] > 0:
            page_confidence += 0.1

        max_confidence = max(max_confidence, min(page_confidence, 1.0))

    execution_time = time.time() - start_time

    return {
        "language": language,
        "page_features": page_features,
        "file_features": {
            "total_pages": len(doc),
            "total_boreholes": total_boreholes,
            "total_descriptions": total_descriptions,
            "total_strip_logs": total_strip_logs,
            "total_tables": total_tables,
            "pages_with_boreholes": pages_with_boreholes,
            "pages_with_names": pages_with_names,
            "pages_with_sidebar": pages_with_sidebar,
            "max_borehole_confidence": max_confidence,
            "has_borehole_page": max_confidence > 0.5,
            "execution_time_seconds": execution_time,
        },
    }
