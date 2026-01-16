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

import rtree
import time

from extraction.features.extract import extract_page, extract_sidebar_information
from extraction.features.metadata.borehole_name_extraction import extract_borehole_names
from extraction.features.metadata.metadata import FileMetadata, MetadataInDocument

from swissgeol_doc_processing.geometry.line_detection import extract_lines
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.text.stemmer import find_matching_expressions
from swissgeol_doc_processing.utils.file_utils import read_params
from swissgeol_doc_processing.utils.strip_log_detection import detect_strip_logs
from swissgeol_doc_processing.utils.table_detection import detect_table_structures

load_dotenv()

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Load configuration files
matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")
name_detection_params = read_params("name_detection_params.yml")
striplog_detection_params = read_params("striplog_detection_params.yml")
table_detection_params = read_params("table_detection_params.yml")


def extract_page_features(
    page: pymupdf.Page,
    page_index: int,
    language: str,
    matching_params: dict,
    line_detection_params: dict,
    name_detection_params: dict,
) -> dict:
    """Extract features from a single page for borehole identification.

    Args:
        page (pymupdf.Page): The PDF page to process.
        page_index (int): The index of the page.
        language (str): The detected language of the document.
        matching_params (dict): Parameters for material description matching.
        line_detection_params (dict): Parameters for line detection.
        name_detection_params (dict): Parameters for borehole name detection.

    Returns:
        dict: Dictionary containing extracted features:
            - number_of_valid_borehole_descriptions: Count of valid material descriptions
            - number_of_strip_logs: Count of detected strip logs
            - number_of_tables: Count of detected tables
            - number_of_boreholes: Count of detected boreholes
            - has_sidebar: Boolean indicating depth column presence
            - borehole_name_entries: List of detected borehole names with confidence scores
            - number_of_grid_lines: Count of horizontal/vertical grid lines
            - number_of_none_gridlines: Count of non-grid geometric lines
            - layer_count: Number of potential layers identified
            - text_line_count: Number of text lines on page
            - borehole_confidence: Overall confidence score (0-1)
    """
    # Extract text and geometric lines
    text_lines = extract_text_lines(page)
    long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)

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

    # Extract strip logs
    strip_logs = detect_strip_logs(page, text_lines, striplog_detection_params)
    number_of_strip_logs = len(strip_logs)

    # Table structures
    table_structures = detect_table_structures(
        page, long_or_horizontal_lines, text_lines, table_detection_params
    )
    number_of_tables = len(table_structures)

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

    # # Detect sidebars (depth columns) - combine all sidebar types
    # # Extract all words from text lines
    # words = [word for line in text_lines for word in line.words]

    # # Build R-tree for spatial queries (for text lines, not geometric lines)
    # line_rtree = rtree.index.Index()
    # for line in text_lines:
    #     line_rtree.insert(id(line), (line.rect.x0, line.rect.y0, line.rect.x1, line.rect.y1), obj=line)

    # # Extract different types of sidebars
    # sidebars: list[Sidebar] = []

    # # Try spulprobe sidebars
    # spulprobe_sidebars = SpulprobeSidebarExtractor.find_in_lines(text_lines)
    # sidebars.extend(spulprobe_sidebars)
    # used_entry_rects = {entry.rect for sidebar in spulprobe_sidebars for entry in sidebar.entries}

    # # Try A-to-B sidebars
    # a_to_b_sidebars = AToBSidebarExtractor.find_in_words(words)
    # sidebars.extend(a_to_b_sidebars)
    # for column in a_to_b_sidebars:
    #     for entry in column.entries:
    #         used_entry_rects.add(entry.start.rect)
    #         used_entry_rects.add(entry.end.rect)

    # # Try A-above-B sidebars
    # a_above_b_sidebars = AAboveBSidebarExtractor.find_in_words(
    #     words, line_rtree, list(used_entry_rects), sidebar_params=matching_params.get("depth_column_params", {})
    # )
    # sidebars.extend([sidebar_noise.sidebar for sidebar_noise in a_above_b_sidebars])

    # # Try layer identifier sidebars
    # layer_identifier_sidebars = LayerIdentifierSidebarExtractor.from_lines(text_lines)
    # sidebars.extend(layer_identifier_sidebars)

    # has_sidebar = len(sidebars) > 0

    # Calculate borehole confidence score (simple heuristic)
    confidence_score = 0.0

    # Each factor contributes to confidence
    if sidebar_information["number_of_good_sidebars"] > 0:
        confidence_score += 0.4
    if number_of_valid_borehole_descriptions > 0:
        confidence_score += min(0.2, number_of_valid_borehole_descriptions * 0.05)
    if number_of_strip_logs > 0:
        confidence_score += 0.3

    confidence_score = min(1.0, confidence_score)

    return {
        "page_number": page_index,
        "number_of_valid_borehole_descriptions": number_of_valid_borehole_descriptions,
        "number_of_strip_logs": number_of_strip_logs,
        "number_of_tables": number_of_tables,
        "number_of_boreholes": number_of_boreholes,
        "sidebar_information": sidebar_information,
        "number_long_or_horizontal_lines": long_or_horizontal_lines,
        "number_all_geometric_lines": all_geometric_lines,
        "text_line_count": len(text_lines),
        "borehole_name_entries": borehole_name_entries,
        "borehole_confidence": confidence_score,
    }


def extract_borehole_features(doc: pymupdf.Document) -> dict:
    """Extract borehole identification features from a PDF document.

    This function processes a PDF document and extracts features for each page
    that can be used as inputs for XGBoost classification.

    Args:
        doc (pymupdf.Document): The opened PDF document to process.

    Returns:
        dict: Dictionary containing:
            - pages: List of per-page feature dictionaries
            - file_features: Aggregated file-level features including:
                - max_borehole_confidence: Highest confidence across all pages
                - has_borehole_page: Boolean if any page has confidence > 0.5
                - total_pages_with_boreholes: Count of pages with confidence > 0.5
                - language: Detected document language
                - has_coordinates: Boolean for coordinate presence
                - has_elevation: Boolean for elevation presence

    Example:
        import pymupdf
        from extraction.minimal_pipeline import extract_borehole_features

        with pymupdf.Document("borehole.pdf") as doc:
            features = extract_borehole_features(doc)
            print(f"Borehole confidence: {features['file_features']['max_borehole_confidence']}")
    """
    start_time = time.time()
    # Extract file metadata
    file_metadata = FileMetadata.from_document(doc, matching_params)
    metadata = MetadataInDocument.from_document(doc, file_metadata.language, matching_params)

    # Extract features from each page
    pages_features = []
    for page_index, page in enumerate(doc):
        page_features = extract_page_features(
            page=page,
            page_index=page_index,
            language=file_metadata.language,
            matching_params=matching_params,
            line_detection_params=line_detection_params,
            name_detection_params=name_detection_params,
        )
        pages_features.append(page_features)

    # Calculate file-level features
    max_borehole_confidence = max((page["borehole_confidence"] for page in pages_features), default=0.0)
    has_borehole_page = max_borehole_confidence > 0.5
    total_pages_with_boreholes = sum(1 for page in pages_features if page["borehole_confidence"] > 0.5)

    execution_time = time.time() - start_time

    return {
        "pages": pages_features,
        "file_features": {
            "max_borehole_confidence": max_borehole_confidence,
            "has_borehole_page": has_borehole_page,
            "total_pages_with_boreholes": total_pages_with_boreholes,
            "language": file_metadata.language,
            "has_coordinates": len(metadata.coordinates) > 0,
            "has_elevation": len(metadata.elevations) > 0,
            "execution_time_seconds": execution_time,
        },
    }
