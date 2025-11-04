"""Swiss Geological Survey Document Processing Library.

A Python library for processing geological borehole documents, including
text extraction, geometric analysis, and document structure detection.
"""

__version__ = "1.0.0"  # Or read from VERSION file

# Public API exports
from .geometry import (
    circle_detection,
    geometric_line_utilities,
    geometry_dataclasses,
    line_detection,
)
from .text import (
    extract_text,
    find_description,
    textblock,
    textline,
)
from .utils import (
    language_detection,
    table_detection,
)

__all__ = [
    "circle_detection",
    "geometric_line_utilities",
    "geometry_dataclasses",
    "line_detection",
    "extract_text",
    "find_description",
    "textblock",
    "textline",
    "language_detection",
    "table_detection",
]
