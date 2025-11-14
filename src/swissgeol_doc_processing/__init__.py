"""Swiss Geological Survey Document Processing Library.

A Python library for processing geological borehole documents, including
text extraction, geometric analysis, and document structure detection.

Instructions:
- usage: from swissgeol_doc_processing.geometry import circle_detection
- usage: from swissgeol_doc_processing.text import extract_text
- usage: from swissgeol_doc_processing.utils import language_detection

List of modules:
- geometry
    - geometric_line_utilities
    - geometry_dataclasses
    - line_detection
    - linesquadtree
    - util
- text
    - extract_text
    - find_description
    - matching_params_analytics
    - stemmer
    - textblock
    - textline_affinity
    - textline
- utils
    - data_extractor
    - file_utils
    - language_detection
    - language_filtering
    - strip_log_detection
    - table_detection
    - utility
"""

# Public API exports
from .geometry import (
    geometric_line_utilities,
    geometry_dataclasses,
    line_detection,
    linesquadtree,
    util,
)
from .text import (
    extract_text,
    find_description,
    matching_params_analytics,
    stemmer,
    textblock,
    textline,
    textline_affinity,
)
from .utils import (
    data_extractor,
    file_utils,
    language_detection,
    language_filtering,
    strip_log_detection,
    table_detection,
    utility,
)

__all__ = [
    "circle_detection",
    "geometric_line_utilities",
    "geometry_dataclasses",
    "line_detection",
    "linesquadtree",
    "util",
    "extract_text",
    "find_description",
    "matching_params_analytics",
    "stemmer",
    "textblock",
    "textline_affinity",
    "textline",
    "data_extractor",
    "file_utils",
    "language_detection",
    "language_filtering",
    "strip_log_detection",
    "table_detection",
    "utility",
]
