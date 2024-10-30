"""Tests for the groundwater module."""

from stratigraphy.evaluation.evaluation_dataclasses import Metrics
from stratigraphy.evaluation.groundwater_evaluator import (
    GroundwaterEvaluator,
    GroundwaterMetrics,
    OverallGroundwaterMetrics,
)
from stratigraphy.groundwater.groundwater_extraction import Groundwater, GroundwaterInDocument


def test_add_groundwater_metrics():
    """Test adding GroundwaterMetrics to OverallGroundwaterMetrics."""
    overall_metrics = OverallGroundwaterMetrics()
    gw_metrics = GroundwaterMetrics(
        groundwater_metrics=Metrics(tp=3, fn=2, fp=1),
        groundwater_depth_metrics=Metrics(tp=3, fn=2, fp=1),
        groundwater_elevation_metrics=Metrics(tp=3, fn=2, fp=1),
        groundwater_date_metrics=Metrics(tp=3, fn=2, fp=1),
        filename="test_file_1",
    )
    overall_metrics.add_groundwater_metrics(gw_metrics)
    assert len(overall_metrics.groundwater_metrics) == 1
    assert overall_metrics.groundwater_metrics[0].filename == "test_file_1"


def test_groundwater_metrics_to_overall_metrics():
    """Test conversion of groundwater metrics to OverallMetrics."""
    overall_metrics = OverallGroundwaterMetrics()
    gw_metrics1 = GroundwaterMetrics(groundwater_metrics=Metrics(tp=3, fn=2, fp=1), filename="file1")
    gw_metrics2 = GroundwaterMetrics(groundwater_metrics=Metrics(tp=3, fn=2, fp=1), filename="file2")
    overall_metrics.add_groundwater_metrics(gw_metrics1)
    overall_metrics.add_groundwater_metrics(gw_metrics2)
    overall = overall_metrics.groundwater_metrics_to_overall_metrics()
    assert "file1" in overall.metrics
    assert "file2" in overall.metrics
    assert overall.metrics["file1"] == gw_metrics1.groundwater_metrics
    assert overall.metrics["file2"] == gw_metrics2.groundwater_metrics


def test_groundwater_depth_metrics_to_overall_metrics():
    """Test conversion of groundwater depth metrics to OverallMetrics."""
    overall_metrics = OverallGroundwaterMetrics()
    gw_metrics = GroundwaterMetrics(groundwater_depth_metrics=Metrics(tp=3, fn=2, fp=1), filename="file_depth")
    overall_metrics.add_groundwater_metrics(gw_metrics)
    overall = overall_metrics.groundwater_depth_metrics_to_overall_metrics()
    assert "file_depth" in overall.metrics
    assert overall.metrics["file_depth"] == gw_metrics.groundwater_depth_metrics


def test_evaluate_with_ground_truth():
    """Test the evaluate method with available ground truth data."""
    ############################################################################################################
    ### Test the from_json_values method of the Groundwater class.
    ############################################################################################################

    # Sample groundwater entries
    groundwater_entries = [
        GroundwaterInDocument(
            filename="example_borehole_profile.pdf",
            groundwater=[Groundwater.from_json_values(depth=2.22, date="2016-04-18", elevation=448.07)],
        )
    ]

    evaluator = GroundwaterEvaluator(groundwater_entries, "example/example_gw_groundtruth.json")
    overall_metrics = evaluator.evaluate()

    # Assertions
    assert isinstance(overall_metrics, OverallGroundwaterMetrics)
    assert len(overall_metrics.groundwater_metrics) == 1
    assert overall_metrics.groundwater_metrics[0].filename == "example_borehole_profile.pdf"
    assert overall_metrics.groundwater_metrics[0].groundwater_metrics.precision == 1.0

    ############################################################################################################
    ### Test the from_json method of the Groundwater class.
    ############################################################################################################

    # Sample groundwater entries
    groundwater_entries = [
        GroundwaterInDocument(
            filename="example_borehole_profile.pdf",
            groundwater=[Groundwater.from_json({"depth": 2.22, "date": "2016-04-18", "elevation": 448.07})],
        )
    ]

    evaluator = GroundwaterEvaluator(groundwater_entries, "example/example_gw_groundtruth.json")
    overall_metrics = evaluator.evaluate()

    # Assertions
    assert isinstance(overall_metrics, OverallGroundwaterMetrics)
    assert len(overall_metrics.groundwater_metrics) == 1
    assert overall_metrics.groundwater_metrics[0].filename == "example_borehole_profile.pdf"
    assert overall_metrics.groundwater_metrics[0].groundwater_metrics.precision == 1.0


def test_evaluate_multiple_entries():
    """Test the evaluate method with multiple groundwater entries."""
    # Sample groundwater entries
    groundwater_entries = [
        GroundwaterInDocument(
            filename="example_borehole_profile.pdf",
            groundwater=[
                Groundwater.from_json_values(depth=2.22, date="2016-04-18", elevation=448.07),
                Groundwater.from_json_values(depth=3.22, date="2016-04-20", elevation=447.07),
            ],
        ),
        GroundwaterInDocument(
            filename="example_borehole_profile_2.pdf",
            groundwater=[Groundwater.from_json_values(depth=3.22, date="2016-04-20", elevation=447.07)],
        ),
    ]

    evaluator = GroundwaterEvaluator(groundwater_entries, "example/example_gw_groundtruth.json")
    overall_metrics = evaluator.evaluate()

    # Assertions
    assert len(overall_metrics.groundwater_metrics) == 2
    assert overall_metrics.groundwater_metrics[0].filename == "example_borehole_profile.pdf"
    assert overall_metrics.groundwater_metrics[1].filename == "example_borehole_profile_2.pdf"
    assert overall_metrics.groundwater_metrics[0].groundwater_metrics.f1 == 1.0
    assert overall_metrics.groundwater_metrics[1].groundwater_metrics.tp == 1.0
    assert overall_metrics.groundwater_metrics[1].groundwater_metrics.fn == 1.0
    assert overall_metrics.groundwater_metrics[1].groundwater_metrics.fp == 0.0
