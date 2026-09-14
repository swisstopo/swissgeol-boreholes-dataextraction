"""Test suite for merge boreholes in continuation detection."""

import pymupdf

from extraction.features.metadata.borehole_name_extraction import BoreholeName
from extraction.features.stratigraphy.layer.continuation_detection import merge_boreholes
from extraction.features.stratigraphy.layer.layer import (
    ExtractedBorehole,
    Layer,
    LayerDepths,
    LayerDepthsEntry,
)
from extraction.features.stratigraphy.layer.overlap_detection import OverlapResult
from extraction.features.stratigraphy.layer.page_bounding_boxes import PageBoundingBoxes
from swissgeol_doc_processing.geometry.geometry_dataclasses import BoundingBox
from swissgeol_doc_processing.text.textblock import MaterialDescription
from swissgeol_doc_processing.utils.data_extractor import FeatureOnPage


def _depth(value: float, page: int) -> LayerDepthsEntry:
    return LayerDepthsEntry(value=value, rect=None, page_number=page)


def _name(text: str, confidence: float = 1.0) -> FeatureOnPage[BoreholeName]:
    return FeatureOnPage(
        feature=BoreholeName(name=text, confidence=confidence), rect=pymupdf.Rect(0, 0, 10, 10), page=1
    )


def _page_bbox(page: int) -> PageBoundingBoxes:
    return PageBoundingBoxes(
        sidebar_bbox=None,
        depth_column_entry_bboxes=[],
        material_description_bbox=BoundingBox(pymupdf.Rect(0, 0, 10, 10)),
        page=page,
    )


def _layer(text: str, start: float | None, end: float | None, page: int) -> Layer:
    return Layer(
        material_description=MaterialDescription(text=text, lines=[]),
        depths=LayerDepths(
            start=_depth(start, page) if start is not None else None,
            end=_depth(end, page) if end is not None else None,
        ),
    )


def test_merge_boreholes_reconciles_duplicated_boundary_layer(monkeypatch):
    """Test that merge_boreholes correctly reconciles duplicated boundary layers.

    Values inspired by 1378_EWS_Salenstein from Thurgau.
    """
    previous_borehole = ExtractedBorehole(
        predictions=[
            _layer("grauer toniger Silt", 64.0, 92.0, 1),
            _layer("grauer siltiger Ton", 92.0, 100.0, 1),
        ],
        bounding_boxes=["bbox_page_1"],
    )

    current_borehole = ExtractedBorehole(
        predictions=[
            _layer("grauer siltiger Ton", None, 108.0, 2),
            _layer("grauer siltiger Ton, wenig Kies", 108.0, 140.0, 2),
        ],
        bounding_boxes=["bbox_page_2"],
    )

    def mock_select_boreholes_with_overlap(previous_page_boreholes, current_page_boreholes, matching_params):
        if previous_page_boreholes == [previous_borehole] and current_page_boreholes == [current_borehole]:
            return previous_borehole, current_borehole, OverlapResult(2, 1)
        return None, None, None

    monkeypatch.setattr(
        "extraction.features.stratigraphy.layer.continuation_detection.select_boreholes_with_overlap",
        mock_select_boreholes_with_overlap,
    )

    merged_boreholes = merge_boreholes(
        boreholes_per_page=[[previous_borehole], [current_borehole]],
        matching_params={},
    )

    assert len(merged_boreholes) == 1

    merged = merged_boreholes[0]
    assert len(merged.predictions) == 3

    assert merged.predictions[0].material_description.text == "grauer toniger Silt"
    assert merged.predictions[0].depths.start.value == 64.0
    assert merged.predictions[0].depths.end.value == 92.0

    assert merged.predictions[1].material_description.text == "grauer siltiger Ton"
    assert merged.predictions[1].depths.start.value == 92.0
    assert merged.predictions[1].depths.end.value == 108.0

    assert merged.predictions[2].material_description.text == "grauer siltiger Ton, wenig Kies"
    assert merged.predictions[2].depths.start.value == 108.0
    assert merged.predictions[2].depths.end.value == 140.0


def test_merge_boreholes_does_not_merge_when_neither_side_has_depths(monkeypatch):
    """Without any depth values to compare, two boreholes must not be merged.

    Regression test: previously, when depths couldn't be compared, the continuation check
    defaulted to accepting the merge, welding unrelated boreholes together (issue seen in 3384.pdf).
    """
    previous_borehole = ExtractedBorehole(
        predictions=[_layer("Humus, Sand", None, None, 1)],
        bounding_boxes=[_page_bbox(1)],
    )
    current_borehole = ExtractedBorehole(
        predictions=[_layer("Humus, Kies", None, None, 2)],
        bounding_boxes=[_page_bbox(2)],
    )

    monkeypatch.setattr(
        "extraction.features.stratigraphy.layer.continuation_detection.select_boreholes_with_overlap",
        lambda previous_page_boreholes, current_page_boreholes, matching_params: (None, None, None),
    )

    merged_boreholes = merge_boreholes(
        boreholes_per_page=[[previous_borehole], [current_borehole]],
        matching_params={},
    )

    assert len(merged_boreholes) == 2


def test_merge_boreholes_name_conflict_blocks_depth_fallback_merge(monkeypatch):
    """Test name conflict.

    A confidently detected, disagreeing name on both sides must block a merge, even when the
    depth-continuity fallback heuristic would otherwise merge them.
    """
    previous_borehole = ExtractedBorehole(
        predictions=[_layer("Sandstein", 10.0, 20.0, 1)],
        bounding_boxes=[_page_bbox(1)],
        name=_name("KB12"),
    )
    current_borehole = ExtractedBorehole(
        predictions=[_layer("Mergel", 20.0, 30.0, 2)],
        bounding_boxes=[_page_bbox(2)],
        name=_name("KB13"),
    )

    monkeypatch.setattr(
        "extraction.features.stratigraphy.layer.continuation_detection.select_boreholes_with_overlap",
        lambda previous_page_boreholes, current_page_boreholes, matching_params: (None, None, None),
    )

    merged_boreholes = merge_boreholes(
        boreholes_per_page=[[previous_borehole], [current_borehole]],
        matching_params={},
    )

    assert len(merged_boreholes) == 2


def test_merge_boreholes_low_confidence_name_does_not_block_merge(monkeypatch):
    """A disagreeing name must not block a merge if either side's detection is low-confidence."""
    previous_borehole = ExtractedBorehole(
        predictions=[_layer("Sandstein", 10.0, 20.0, 1)],
        bounding_boxes=[_page_bbox(1)],
        name=_name("KB12", confidence=0.2),
    )
    current_borehole = ExtractedBorehole(
        predictions=[_layer("Mergel", 20.0, 30.0, 2)],
        bounding_boxes=[_page_bbox(2)],
        name=_name("KB13"),
    )

    monkeypatch.setattr(
        "extraction.features.stratigraphy.layer.continuation_detection.select_boreholes_with_overlap",
        lambda previous_page_boreholes, current_page_boreholes, matching_params: (None, None, None),
    )

    merged_boreholes = merge_boreholes(
        boreholes_per_page=[[previous_borehole], [current_borehole]],
        matching_params={},
    )

    assert len(merged_boreholes) == 1
