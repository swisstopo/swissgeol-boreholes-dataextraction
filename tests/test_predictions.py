"""Test suite for the prediction module."""

from pathlib import Path
from unittest.mock import Mock

import fitz
import pytest
from stratigraphy.metadata.coordinate_extraction import CoordinateEntry, LV95Coordinate
from stratigraphy.metadata.metadata import BoreholeMetadata
from stratigraphy.util.predictions import FilePredictions, OverallFilePredictions

# Mock classes used in the FilePredictions constructor
LayerPrediction = Mock()
GroundwaterInformationOnPage = Mock()
DepthsMaterialsColumnPairs = Mock()


@pytest.fixture
def sample_file_prediction():
    """Fixture to create a sample FilePredictions object."""
    coord = LV95Coordinate(
        east=CoordinateEntry(coordinate_value=2789456),
        north=CoordinateEntry(coordinate_value=1123012),
        rect=fitz.Rect(),
        page=1,
    )

    layer1 = Mock(
        material_description=Mock(text="Sand"), depth_interval=Mock(start=Mock(value=10), end=Mock(value=20))
    )
    layer2 = Mock(
        material_description=Mock(text="Clay"), depth_interval=Mock(start=Mock(value=30), end=Mock(value=50))
    )
    metadata = BoreholeMetadata(coordinates=coord, page_dimensions=[Mock(width=10, height=20)], language="en")

    return FilePredictions(
        layers=[layer1, layer2],
        file_name="test_file",
        metadata=metadata,
        groundwater=None,
        depths_materials_columns_pairs=None,
    )


def test_convert_to_ground_truth(sample_file_prediction):
    """Test the convert_to_ground_truth method."""
    ground_truth = sample_file_prediction.convert_to_ground_truth()

    assert ground_truth["test_file"]["metadata"]["coordinates"]["E"] == 2789456
    assert ground_truth["test_file"]["metadata"]["coordinates"]["N"] == 1123012
    assert len(ground_truth["test_file"]["layers"]) == 2
    assert ground_truth["test_file"]["layers"][0]["material_description"] == "Sand"


def test_to_json(sample_file_prediction):
    """Test the to_json method."""
    result = sample_file_prediction.to_json()

    assert isinstance(result, dict)
    assert result["file_name"] == "test_file"
    assert len(result["layers"]) == 2
    assert result["metadata"]["coordinates"]["E"] == 2789456


def test_count_against_ground_truth():
    """Test the count_against_ground_truth static method."""
    values = [1, 2, 2, 3]
    ground_truth = [2, 3, 4]

    # TODO: This is deprecated, and should be removed
    metrics = FilePredictions.count_against_ground_truth(values, ground_truth)

    assert metrics.tp == 2
    assert metrics.fp == 2
    assert metrics.fn == 1


def test_overall_file_predictions():
    """Test OverallFilePredictions class functionality."""
    overall_predictions = OverallFilePredictions()
    file_prediction = Mock(to_json=lambda: {"some_data": "test"}, file_name="test_file")

    overall_predictions.add_file_predictions(file_prediction)
    result = overall_predictions.to_json()

    assert len(result) == 1
    assert result == {"test_file": {"some_data": "test"}}


def test_evaluate_groundwater(sample_file_prediction):
    """Test the evaluate_groundwater method."""
    sample_file_prediction.groundwater = [
        Mock(groundwater=Mock(depth=100, format_date=lambda: "2024-10-01", elevation=20))
    ]
    groundwater_gt = [{"depth": 100, "date": "2024-10-01", "elevation": 20}]

    # TODO: This is deprecated, and should be removed
    sample_file_prediction.evaluate_groundwater(groundwater_gt)

    assert sample_file_prediction.groundwater_is_correct["groundwater"].tp == 1
    assert sample_file_prediction.groundwater_is_correct["groundwater_depth"].tp == 1


def test_evaluate_metadata_extraction():
    """Test evaluate_metadata_extraction method of OverallFilePredictions."""
    overall_predictions = OverallFilePredictions()
    file_prediction = Mock(metadata=Mock(to_json=lambda: {"coordinates": "some_coordinates"}))
    overall_predictions.add_file_predictions(file_prediction)

    ground_truth_path = Path("example/example_groundtruth.json")
    metadata_metrics = overall_predictions.evaluate_metadata_extraction(ground_truth_path)

    assert metadata_metrics is not None  # Ensure the evaluation returns a result
