"""Tests for the groundwater module."""

import pytest
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.evaluation.groundwater_evaluator import (
    GroundwaterEvaluator,
    GroundwaterMetrics,
    OverallGroundwaterMetrics,
)
from stratigraphy.groundwater.groundwater_extraction import Groundwater, GroundwaterInDocument


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return Metrics(tp=3, fn=2, fp=1)


@pytest.fixture
def groundtruth_path():
    """Path to the ground truth file."""
    return "example/example_gw_groundtruth.json"


def test_add_groundwater_metrics(sample_metrics):
    """Test adding GroundwaterMetrics to OverallGroundwaterMetrics."""
    overall_metrics = OverallGroundwaterMetrics()
    gw_metrics = GroundwaterMetrics(
        groundwater_metrics=sample_metrics,
        groundwater_depth_metrics=sample_metrics,
        groundwater_elevation_metrics=sample_metrics,
        groundwater_date_metrics=sample_metrics,
        filename="test_file_1",
    )
    overall_metrics.add_groundwater_metrics(gw_metrics)
    assert len(overall_metrics.groundwater_metrics) == 1
    assert overall_metrics.groundwater_metrics[0].filename == "test_file_1"


def test_groundwater_metrics_to_overall_metrics(sample_metrics):
    """Test conversion of groundwater metrics to OverallMetrics."""
    overall_metrics = OverallGroundwaterMetrics()
    gw_metrics1 = GroundwaterMetrics(groundwater_metrics=sample_metrics, filename="file1")
    gw_metrics2 = GroundwaterMetrics(groundwater_metrics=sample_metrics, filename="file2")
    overall_metrics.add_groundwater_metrics(gw_metrics1)
    overall_metrics.add_groundwater_metrics(gw_metrics2)
    overall = overall_metrics.groundwater_metrics_to_overall_metrics()
    assert "file1" in overall.metrics
    assert "file2" in overall.metrics
    assert overall.metrics["file1"] == gw_metrics1.groundwater_metrics
    assert overall.metrics["file2"] == gw_metrics2.groundwater_metrics


def test_groundwater_depth_metrics_to_overall_metrics(sample_metrics):
    """Test conversion of groundwater depth metrics to OverallMetrics."""
    overall_metrics = OverallGroundwaterMetrics()
    gw_metrics = GroundwaterMetrics(groundwater_depth_metrics=sample_metrics, filename="file_depth")
    overall_metrics.add_groundwater_metrics(gw_metrics)
    overall = overall_metrics.groundwater_depth_metrics_to_overall_metrics()
    assert "file_depth" in overall.metrics
    assert overall.metrics["file_depth"] == gw_metrics.groundwater_depth_metrics


def test_evaluate_with_ground_truth(groundtruth_path):
    """Test the evaluate method with available ground truth data."""
    # Sample groundwater entries
    groundwater_entries = [
        GroundwaterInDocument(
            filename="example_borehole_profile.pdf",
            groundwater=[
                FeatureOnPage.from_json(
                    {"depth": 2.22, "date": "2016-04-18", "elevation": 448.07, "page": 1, "rect": [0, 0, 100, 100]},
                    Groundwater,
                )
            ],
        )
    ]

    evaluator = GroundwaterEvaluator(groundwater_entries, groundtruth_path)
    overall_metrics = evaluator.evaluate()

    # Assertions
    assert isinstance(overall_metrics, OverallGroundwaterMetrics)
    assert len(overall_metrics.groundwater_metrics) == 1
    assert overall_metrics.groundwater_metrics[0].filename == "example_borehole_profile.pdf"
    assert overall_metrics.groundwater_metrics[0].groundwater_metrics.precision == 1.0


def test_evaluate_multiple_entries(groundtruth_path):
    """Test the evaluate method with multiple groundwater entries."""
    # Sample groundwater entries
    groundwater_entries = [
        GroundwaterInDocument(
            filename="example_borehole_profile.pdf",
            groundwater=[
                FeatureOnPage.from_json(
                    {"depth": 2.22, "date": "2016-04-18", "elevation": 448.07, "page": 1, "rect": [0, 0, 100, 100]},
                    Groundwater,
                ),
                FeatureOnPage.from_json(
                    {"depth": 3.22, "date": "2016-04-20", "elevation": 447.07, "page": 1, "rect": [0, 0, 100, 100]},
                    Groundwater,
                ),
            ],
        ),
        GroundwaterInDocument(
            filename="example_borehole_profile_2.pdf",
            groundwater=[
                FeatureOnPage.from_json(
                    {"depth": 3.22, "date": "2016-04-20", "elevation": 447.07, "page": 1, "rect": [0, 0, 100, 100]},
                    Groundwater,
                )
            ],
        ),
    ]

    evaluator = GroundwaterEvaluator(groundwater_entries, groundtruth_path)
    overall_metrics = evaluator.evaluate()

    # Assertions
    assert len(overall_metrics.groundwater_metrics) == 2
    assert overall_metrics.groundwater_metrics[0].filename == "example_borehole_profile.pdf"
    assert overall_metrics.groundwater_metrics[1].filename == "example_borehole_profile_2.pdf"
    assert overall_metrics.groundwater_metrics[0].groundwater_metrics.f1 == 1.0
    assert overall_metrics.groundwater_metrics[1].groundwater_metrics.tp == 1.0
    assert overall_metrics.groundwater_metrics[1].groundwater_metrics.fn == 1.0
    assert overall_metrics.groundwater_metrics[1].groundwater_metrics.fp == 0.0


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"depth": 2.22, "date": "2016-04-18", "elevation": 448.07}, True),  # Valid case
        ({"depth": -1, "date": "2016-04-18", "elevation": 448.07}, False),  # Invalid depth
    ],
)
def test_groundwater_validation(test_input, expected):
    """Test the is_valid method of the Groundwater class."""
    groundwater = Groundwater.from_json(test_input)
    assert groundwater.is_valid() == expected
