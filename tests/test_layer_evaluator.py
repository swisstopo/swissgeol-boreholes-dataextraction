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
from swissgeol_doc_processing.text.textblock import MaterialDescription, MaterialDescriptionLine


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


@pytest.fixture
def test_case_3():
    """Test fixture for case where depths are correct and material descriptions are 'of by one'.

    Depending on the score use, the matching won't be the same.
    """
    preds = [
        create_test_layer("B", start=0.0, end=1.0),
        create_test_layer("C", start=1.0, end=2.0),
        create_test_layer("D", start=2.0, end=3.0),
    ]

    gt = [
        {"material_description": "A", "depth_interval": {"start": 0.0, "end": 1.0}},
        {"material_description": "B", "depth_interval": {"start": 1.0, "end": 2.0}},
        {"material_description": "C", "depth_interval": {"start": 2.0, "end": 3.0}},
    ]

    return gt, preds


@pytest.fixture
def test_case_4():
    """Text fixture for case where depths are correct but material descriptions are 'off by one'.

    In this scenario, both depth-based and material-based mappings yield a score of 0.66.
    When using the combined layer score, there is a tie: either a direct alignment (favoring depth) or an off-by-one
    alignment (favoring material). The mapping resolves the tie by preferring the alignment that maximizes the number
    of matched layers, in this case, the depth-preserving alignment, which maps 3 layers instead of 2.
    """
    preds = [
        create_test_layer("B", start=0.0, end=1.0),
        create_test_layer("C", start=1.0, end=2.0),
        create_test_layer("D", start=2.0, end=3.0),
    ]

    gt = [
        {"material_description": "A", "depth_interval": {"start": 0.0, "end": 1.0}},
        {"material_description": "B", "depth_interval": {"start": 1.0, "end": 2.0}},
        {"material_description": "C", "depth_interval": {"start": 24.0, "end": 35.0}},
    ]

    return gt, preds


@pytest.mark.parametrize(
    "test_case,scoring_fn,expected_mapping,expected_score",
    [
        pytest.param("test_case_1", score_layer, [(0, 0), (1, 1), (2, 2)], 0.5, id="order_preserving_match"),
        pytest.param("test_case_2", score_layer, [(2, 0)], 1 / 3, id="perfect_depth_match"),
        pytest.param("test_case_3", score_layer, [(0, 0), (1, 1), (2, 2)], 0.5, id="direct_layer_mapping"),
        pytest.param("test_case_3", score_depths, [(0, 0), (1, 1), (2, 2)], 1.0, id="direct_depth_mapping"),
        pytest.param("test_case_3", score_material_descriptions, [(0, 1), (1, 2)], 2 / 3, id="off_by_one_mat_mapping"),
        pytest.param("test_case_4", score_layer, [(0, 0), (1, 1), (2, 2)], 1 / 3, id="layer_mapping_score_tie"),
    ],
)
def test_layer_matching(test_case, scoring_fn, expected_mapping, expected_score, request):
    """Test layer matching for different scenarios."""
    gt, preds = request.getfixturevalue(test_case)

    score, mapping = LayerEvaluator.compute_borehole_affinity_and_mapping(gt, preds, scoring_fn)

    expected_pairs = [(preds[i], gt[j]) for i, j in expected_mapping]
    assert mapping == expected_pairs, f"expected mapping {expected_mapping}, got {mapping}"
    assert pytest.approx(score, rel=1e-6) == expected_score
