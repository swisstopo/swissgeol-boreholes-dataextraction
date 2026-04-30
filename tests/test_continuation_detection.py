"""Test suite for merge boreholes in continuation detection."""

from extraction.features.stratigraphy.layer.continuation_detection import merge_boreholes
from extraction.features.stratigraphy.layer.layer import (
    ExtractedBorehole,
    Layer,
    LayerDepths,
    LayerDepthsEntry,
)
from extraction.features.stratigraphy.layer.overlap_detection import OverlapResult
from swissgeol_doc_processing.text.textblock import MaterialDescription


def _depth(value: float, page: int) -> LayerDepthsEntry:
    return LayerDepthsEntry(value=value, rect=None, page_number=page)


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
