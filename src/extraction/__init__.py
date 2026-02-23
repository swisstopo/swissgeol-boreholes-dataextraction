"""Borehole data extraction package.

A Python library for extracting structured data from borehole PDF documents,
including metadata, stratigraphy, and features for machine learning classification.

Instructions:
- usage: from extraction.features import extract
- usage: from extraction.features.metadata import borehole_name_extraction
- usage: from extraction.minimal_pipeline import extract_page_features

List of modules:
- features
    - metadata: Borehole metadata extraction (coordinates, elevation, names)
    - stratigraphy: Layer and depth extraction
    - predictions: Data structures for results
    - extract: Main extraction logic
- evaluation: Evaluation and benchmarking
- annotations: Visualization and drawing
- main: Full extraction pipeline
- minimal_pipeline: Minimal identification pipeline for classification
"""

# Import submodules
from extraction import annotations, evaluation, features, minimal_pipeline

__all__ = ["annotations", "evaluation", "features", "utils", "minimal_pipeline"]
