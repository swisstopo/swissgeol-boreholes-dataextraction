"""Tests for the LayerEvaluator class."""

import pymupdf
import pytest

from extraction.evaluation.layer_evaluator import LayerEvaluator
from extraction.features.stratigraphy.layer.layer import Layer, LayerDepths, LayerDepthsEntry, LayersInBorehole
from extraction.features.utils.text.textblock import MaterialDescription, MaterialDescriptionLine


def create_test_layer(text: str, start: float, end: float) -> Layer:
    """Helper function to create a test layer."""
    return Layer(
        MaterialDescription((text), [MaterialDescriptionLine(text)]),
        LayerDepths(LayerDepthsEntry(start, pymupdf.Rect(), 0), LayerDepthsEntry(end, pymupdf.Rect(), 0)),
    )


@pytest.fixture
def test_case_1():
    """Test fixture for case where match pred[0] -> gt[0] is good enough."""
    preds = LayersInBorehole(
        layers=[
            create_test_layer("C", start=0.0, end=1.0),
            create_test_layer("B", start=1.0, end=2.0),
            create_test_layer("A", start=0.0, end=1.0),
        ]
    )
    gt = [
        {"material_description": "A", "depth_interval": {"start": 0.0, "end": 1.0}},
        {"material_description": "B", "depth_interval": {"start": 1.0, "end": 2.0}},
        {"material_description": "C", "depth_interval": {"start": 2.0, "end": 3.0}},
    ]

    return gt, preds


@pytest.fixture
def test_case_2():
    """Test fixture for case where match gt[0] -> pred[2] is too good (it will be the only match)."""
    preds = LayersInBorehole(
        layers=[
            create_test_layer("C", start=0.0, end=1.5),
            create_test_layer("B", start=1.5, end=2.5),
            create_test_layer("A", start=0.0, end=1.0),
        ]
    )
    gt = [
        {"material_description": "A", "depth_interval": {"start": 0.0, "end": 1.0}},
        {"material_description": "B", "depth_interval": {"start": 1.0, "end": 2.0}},
        {"material_description": "C", "depth_interval": {"start": 2.0, "end": 3.0}},
    ]

    return gt, preds


@pytest.mark.parametrize(
    "test_case,expected_mapping,expected_score",
    [
        pytest.param("test_case_1", [(0, 0), (1, 1), (2, 2)], 1.5, id="order_preserving_match"),
        pytest.param("test_case_2", [(2, 0)], 1.0, id="perfect_depth_match"),
    ],
)
def test_layer_matching(test_case, expected_mapping, expected_score, request):
    """Test layer matching for different scenarios."""
    gt, preds = request.getfixturevalue(test_case)

    score, mapping = LayerEvaluator.compute_borehole_affinity_and_mapping(gt, preds)

    mapped_pairs = [(m["pred_idx"], m["gt_idx"]) for m in mapping]
    assert mapped_pairs == expected_mapping, f"expected mapping {expected_mapping}, got {mapped_pairs}"
    assert pytest.approx(score, rel=1e-6) == expected_score
