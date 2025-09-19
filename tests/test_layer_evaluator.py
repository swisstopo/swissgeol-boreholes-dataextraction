"""Tests for the LayerEvaluator class."""

import pymupdf
import pytest

from extraction.evaluation.layer_evaluator import (
    LayerEvaluator,
    score_depths,
    score_layer,
    score_material_descriptions,
)
from extraction.features.stratigraphy.layer.layer import Layer, LayerDepths, LayerDepthsEntry
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
    preds = [
        create_test_layer("C", start=0.0, end=1.0),
        create_test_layer("B", start=1.0, end=2.0),
        create_test_layer("A", start=0.0, end=1.0),
    ]

    gt = [
        {"material_description": "A", "depth_interval": {"start": 0.0, "end": 1.0}},
        {"material_description": "B", "depth_interval": {"start": 1.0, "end": 2.0}},
        {"material_description": "C", "depth_interval": {"start": 2.0, "end": 3.0}},
    ]

    return gt, preds


@pytest.fixture
def test_case_2():
    """Test fixture for case where match gt[0] -> pred[2] is too good (it will be the only match)."""
    preds = [
        create_test_layer("C", start=0.0, end=1.5),
        create_test_layer("B", start=1.5, end=2.5),
        create_test_layer("A", start=0.0, end=1.0),
    ]

    gt = [
        {"material_description": "A", "depth_interval": {"start": 0.0, "end": 1.0}},
        {"material_description": "B", "depth_interval": {"start": 1.0, "end": 2.0}},
        {"material_description": "C", "depth_interval": {"start": 2.0, "end": 3.0}},
    ]

    return gt, preds


@pytest.mark.parametrize(
    "test_case,expected_mapping,expected_score",
    [
        pytest.param("test_case_1", [(0, 0), (1, 1), (2, 2)], 0.25, id="order_preserving_match"),
        pytest.param("test_case_2", [(2, 0)], 1 / 6, id="perfect_depth_match"),
    ],
)
def test_layer_matching(test_case, expected_mapping, expected_score, request):
    """Test layer matching for different scenarios."""
    gt, preds = request.getfixturevalue(test_case)

    score, mapping = LayerEvaluator.compute_borehole_affinity_and_mapping(gt, preds, score_layer)

    expected_pairs = [(preds[i], gt[j]) for i, j in expected_mapping]
    assert mapping == expected_pairs, f"expected mapping {expected_mapping}, got {mapping}"
    assert pytest.approx(score, rel=1e-6) == expected_score


def test_off_by_one_layer_matching():
    """Check that we get different mappings for depths and material descriptions in case of misaligned predictions."""
    preds = [create_test_layer("B", start=0.0, end=1.0), create_test_layer("C", start=1.0, end=2.0)]

    gt = [
        {"material_description": "A", "depth_interval": {"start": 0.0, "end": 1.0}},
        {"material_description": "B", "depth_interval": {"start": 1.0, "end": 2.0}},
        {"material_description": "C", "depth_interval": {"start": 2.0, "end": 3.0}},
    ]

    _, mapping_depths = LayerEvaluator.compute_borehole_affinity_and_mapping(gt, preds, score_depths)
    expected_pairs = [(preds[i], gt[j]) for i, j in [(0, 0), (1, 1)]]
    assert mapping_depths == expected_pairs, f"expected mapping {expected_pairs}, got {mapping_depths}"

    _, mapping_material_descriptions = LayerEvaluator.compute_borehole_affinity_and_mapping(
        gt, preds, score_material_descriptions
    )
    expected_pairs = [(preds[i], gt[j]) for i, j in [(0, 1), (1, 2)]]
    assert mapping_material_descriptions == expected_pairs, (
        f"expected mapping {expected_pairs}, got {mapping_material_descriptions}"
    )
