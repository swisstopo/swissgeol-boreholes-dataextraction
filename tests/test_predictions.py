"""Test suite for the prediction module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import fitz
import pytest
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.evaluation.utility import count_against_ground_truth
from stratigraphy.groundwater.groundwater_extraction import Groundwater, GroundwaterInDocument
from stratigraphy.layer.layer import LayersInDocument, LayersOnPage
from stratigraphy.metadata.coordinate_extraction import CoordinateEntry, LV95Coordinate
from stratigraphy.metadata.metadata import BoreholeMetadata
from stratigraphy.util.predictions import FilePredictions, OverallFilePredictions


@pytest.fixture
def sample_file_prediction() -> FilePredictions:
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
    layer_on_page = LayersOnPage(layers_on_page=[layer1, layer2])
    layers_in_document = LayersInDocument(layers_in_document=[layer_on_page], filename="test_file")

    dt_date = datetime(2024, 10, 1)
    groundwater_on_page = FeatureOnPage(
        feature=Groundwater(depth=100, date=dt_date, elevation=20),
        page=1,
        rect=fitz.Rect(0, 0, 100, 100),
    )
    groundwater_in_doc = GroundwaterInDocument(groundwater=[groundwater_on_page], filename="test_file")

    metadata = BoreholeMetadata(coordinates=coord, page_dimensions=[Mock(width=10, height=20)], language="en")

    return FilePredictions(
        layers=layers_in_document,
        file_name="test_file",
        metadata=metadata,
        groundwater=groundwater_in_doc,
        depths_materials_columns_pairs=None,
    )


def test_convert_to_ground_truth(sample_file_prediction: FilePredictions):
    """Test the convert_to_ground_truth method."""
    ground_truth = sample_file_prediction.convert_to_ground_truth()

    assert ground_truth["test_file"]["metadata"]["coordinates"]["E"] == 2789456
    assert ground_truth["test_file"]["metadata"]["coordinates"]["N"] == 1123012
    assert len(ground_truth["test_file"]["layers"]) == 2
    assert ground_truth["test_file"]["layers"][0]["material_description"] == "Sand"


def test_to_json(sample_file_prediction: FilePredictions):
    """Test the to_json method."""
    result = sample_file_prediction.to_json()

    assert isinstance(result, dict)
    assert result["file_name"] == "test_file"
    assert len(result["layers"]) == 2
    assert result["metadata"]["coordinates"]["E"] == 2789456


def test_overall_file_predictions():
    """Test OverallFilePredictions class functionality."""
    overall_predictions = OverallFilePredictions()
    file_prediction = Mock(to_json=lambda: {"some_data": "test"}, file_name="test_file")

    overall_predictions.add_file_predictions(file_prediction)
    result = overall_predictions.to_json()

    assert len(result) == 1
    assert result == {"test_file": {"some_data": "test"}}


def test_evaluate_metadata_extraction():
    """Test evaluate_metadata_extraction method of OverallFilePredictions."""
    overall_predictions = OverallFilePredictions()
    file_prediction = Mock(metadata=Mock(to_json=lambda: {"coordinates": "some_coordinates"}))
    overall_predictions.add_file_predictions(file_prediction)

    ground_truth_path = Path("example/example_groundtruth.json")
    metadata_metrics = overall_predictions.evaluate_metadata_extraction(ground_truth_path)

    assert metadata_metrics is not None  # Ensure the evaluation returns a result


@pytest.mark.parametrize(
    "values,ground_truth,expected",
    [
        # Current case
        ([1, 2, 2, 3], [2, 3, 4], (2, 2, 1)),
        # Empty lists
        ([], [], (0, 0, 0)),
        ([], [1, 2], (0, 0, 2)),
        ([1, 2], [], (0, 2, 0)),
        # Exact match
        ([1, 2], [1, 2], (2, 0, 0)),
        # No matches
        ([1, 2], [3, 4], (0, 2, 2)),
    ],
)
def test_count_against_ground_truth_cases(values, ground_truth, expected):
    """Test count_against_ground_truth with various scenarios."""
    metrics = count_against_ground_truth(values, ground_truth)
    assert (metrics.tp, metrics.fp, metrics.fn) == expected
