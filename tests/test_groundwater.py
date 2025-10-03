"""Tests for the groundwater module."""

from datetime import date

import pytest

from extraction.evaluation.benchmark.ground_truth import GroundTruth
from extraction.evaluation.evaluation_dataclasses import Metrics
from extraction.evaluation.groundwater_evaluator import (
    GroundwaterEvaluator,
    GroundwaterMetrics,
    OverallGroundwaterMetrics,
)
from extraction.features.groundwater.groundwater_extraction import Groundwater, GroundwatersInBorehole
from extraction.features.groundwater.utility import extract_date
from extraction.features.predictions.borehole_predictions import (
    BoreholeGroundwaterWithGroundTruth,
    FileGroundwaterWithGroundTruth,
)
from extraction.features.utils.data_extractor import FeatureOnPage


@pytest.fixture
def date_test_cases():
    """Provides test cases for extract_date function."""
    return [
        # Valid full-year dates
        ("The date is 12.05.1998", date(1998, 5, 12), "12.05.1998"),
        # Valid short-year dates
        ("Event on 07.04.99", date(1999, 4, 7), "07.04.99"),
        # Extra spaces
        ("  10 . 03 . 1985 ", date(1985, 3, 10), "10 . 03 . 1985"),
        # Invalid dates
        ("Invalid format: 30.02.2020", None, None),
        # No date in text
        ("No date here!", None, None),
    ]


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return Metrics(tp=3, fn=2, fp=1)


@pytest.fixture
def groundtruth():
    """Path to the ground truth file."""
    return GroundTruth("example/example_gw_groundtruth.json")


@pytest.fixture
def groundwater_at_2m22() -> dict:
    """Fixture that returns an Groundwater object (embeded in a FeatureOnPage)."""
    return FeatureOnPage.from_json(
        {
            "depth": 2.22,
            "date": "2016-04-18",
            "elevation": 448.07,
            "page": 1,
            "rect": [0, 0, 100, 100],
        },
        Groundwater,
    )


@pytest.fixture
def groundwater_at_3m22() -> dict:
    """Fixture that returns another Groundwater object (embeded in a FeatureOnPage)."""
    return FeatureOnPage.from_json(
        {
            "depth": 3.22,
            "date": "2016-04-20",
            "elevation": 447.07,
            "page": 1,
            "rect": [0, 0, 100, 100],
        },
        Groundwater,
    )


def test_extract_date(date_test_cases):
    """Test extract_date function with various inputs."""
    for text, expected_date, expected_str in date_test_cases:
        assert extract_date(text) == (expected_date, expected_str)


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


def test_evaluate_with_ground_truth(groundtruth, groundwater_at_2m22, groundwater_at_3m22):
    """Test the evaluate method with available ground truth data."""
    # In this test, there is one borehole, with two groundwater measurement for it.
    groundwater_entries = {
        "example_borehole_profile.pdf": [GroundwatersInBorehole([groundwater_at_2m22, groundwater_at_3m22])]
    }

    # dictionary used to "manually" build the FileGroundwaterWithGroundTruth object
    pred_to_gt_matching = {"example_borehole_profile.pdf": {0: 0}}
    evaluator = GroundwaterEvaluator(
        groundwater_list=[
            FileGroundwaterWithGroundTruth(
                filename=filename,
                boreholes=[
                    BoreholeGroundwaterWithGroundTruth(
                        groundwater=groundwaterinborehole,
                        ground_truth=groundtruth.for_file(filename).get(pred_to_gt_matching[filename][pred_idx])[
                            "groundwater"
                        ],
                    )
                    for pred_idx, groundwaterinborehole in enumerate(groundwaterinborehole_list)
                ],
            )
            for filename, groundwaterinborehole_list in groundwater_entries.items()
        ]
    )
    overall_metrics = evaluator.evaluate()

    # Assertions
    assert isinstance(overall_metrics, OverallGroundwaterMetrics)
    assert len(overall_metrics.groundwater_metrics) == 1
    assert overall_metrics.groundwater_metrics[0].filename == "example_borehole_profile.pdf"
    assert overall_metrics.groundwater_metrics[0].groundwater_metrics.precision == 1.0


def test_evaluate_multiple_documents(groundtruth, groundwater_at_2m22, groundwater_at_3m22):
    """Test the evaluate method with multiple groundwater entries on two documents.

    On the first document, there is one borehole with two groundwater measurement (like in the previous test).
    On the second document, there is two boreholes, each with one groundwater measurement. For those boreholes, the
    matching with the ground truth is 0->1 and 1->0. Meaning that the first borehole of the prediction matches the
    second entry in the groudtruth file.
    """
    # Sample groundwater entries
    gt_matching_index_example = {0: 0}
    gt_matching_index_example_2 = {0: 1, 1: 0}
    evaluator = GroundwaterEvaluator(
        groundwater_list=[
            FileGroundwaterWithGroundTruth(
                filename="example_borehole_profile.pdf",
                boreholes=[
                    BoreholeGroundwaterWithGroundTruth(
                        groundwater=GroundwatersInBorehole([groundwater_at_2m22, groundwater_at_3m22]),
                        ground_truth=groundtruth.for_file("example_borehole_profile.pdf").get(
                            gt_matching_index_example[0]
                        )["groundwater"],
                    )
                ],
            ),
            FileGroundwaterWithGroundTruth(
                filename="example_borehole_profile_2.pdf",
                boreholes=[
                    BoreholeGroundwaterWithGroundTruth(
                        groundwater=GroundwatersInBorehole([groundwater_at_2m22]),
                        ground_truth=groundtruth.for_file("example_borehole_profile_2.pdf").get(
                            gt_matching_index_example_2[0]
                        )["groundwater"],
                    ),
                    BoreholeGroundwaterWithGroundTruth(
                        groundwater=GroundwatersInBorehole([groundwater_at_3m22]),
                        ground_truth=groundtruth.for_file("example_borehole_profile_2.pdf").get(
                            gt_matching_index_example_2[1]
                        )["groundwater"],
                    ),
                ],
            ),
        ]
    )
    overall_metrics = evaluator.evaluate()

    # Assertions
    assert len(overall_metrics.groundwater_metrics) == 2
    assert overall_metrics.groundwater_metrics[0].filename == "example_borehole_profile.pdf"
    assert overall_metrics.groundwater_metrics[1].filename == "example_borehole_profile_2.pdf"
    assert overall_metrics.groundwater_metrics[0].groundwater_metrics.f1 == 1.0
    assert overall_metrics.groundwater_metrics[1].groundwater_metrics.tp == 2.0
    assert overall_metrics.groundwater_metrics[1].groundwater_metrics.fn == 0.0
    assert overall_metrics.groundwater_metrics[1].groundwater_metrics.fp == 0.0
