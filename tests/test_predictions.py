"""Test suite for the prediction module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import fitz
import pytest
from stratigraphy.benchmark.ground_truth import GroundTruth
from stratigraphy.data_extractor.data_extractor import FeatureOnPage
from stratigraphy.evaluation.utility import count_against_ground_truth
from stratigraphy.groundwater.groundwater_extraction import Groundwater, GroundwatersInBorehole
from stratigraphy.layer.layer import LayersInBorehole
from stratigraphy.metadata.coordinate_extraction import CoordinateEntry, LV95Coordinate
from stratigraphy.metadata.metadata import BoreholeMetadata, FileMetadata
from stratigraphy.util.predictions import BoreholePredictions, FilePredictions, OverallFilePredictions


@pytest.fixture
def sample_file_prediction() -> FilePredictions:
    """Fixture to create a sample FilePredictions object."""
    filename = "example_borehole_profile.pdf"
    coord = FeatureOnPage(
        feature=LV95Coordinate(
            east=CoordinateEntry(coordinate_value=2789456), north=CoordinateEntry(coordinate_value=1123012)
        ),
        rect=fitz.Rect(),
        page=1,
    )

    layer1 = Mock(
        material_description=Mock(text="Sand"), depth_interval=Mock(start=Mock(value=10), end=Mock(value=20))
    )
    layer2 = Mock(
        material_description=Mock(text="Clay"), depth_interval=Mock(start=Mock(value=30), end=Mock(value=50))
    )
    layers_in_borehole = LayersInBorehole(layers=[layer1, layer2])

    dt_date = datetime(2024, 10, 1)
    groundwater_on_page = FeatureOnPage(
        feature=Groundwater(depth=100, date=dt_date, elevation=20),
        page=1,
        rect=fitz.Rect(0, 0, 100, 100),
    )
    groundwater_in_bh = GroundwatersInBorehole(groundwater_feature_list=[groundwater_on_page])

    file_metadata = FileMetadata(language="en", filename=filename, page_dimensions=[Mock(width=10, height=20)])
    metadata = BoreholeMetadata(coordinates=coord, elevation=None)

    return FilePredictions(
        [
            BoreholePredictions(
                borehole_index=0,
                layers_in_borehole=layers_in_borehole,
                file_name=filename,
                metadata=metadata,
                groundwater_in_borehole=groundwater_in_bh,
                bounding_boxes=[],
            )
        ],
        file_name=filename,
        file_metadata=file_metadata,
    )


def test_to_json(sample_file_prediction: FilePredictions):
    """Test the to_json method."""
    result = sample_file_prediction.to_json()

    assert isinstance(result, dict)
    assert len(result["boreholes"][0]["layers"]) == 2
    assert result["boreholes"][0]["metadata"]["coordinates"]["E"] == 2789456
    assert result["language"] == "en"


def test_overall_file_predictions(sample_file_prediction: FilePredictions):
    """Test OverallFilePredictions class functionality."""
    overall_predictions = OverallFilePredictions()

    overall_predictions.add_file_predictions(sample_file_prediction)
    result = overall_predictions.to_json()

    assert len(result) == 1
    assert set(result.keys()) == {"example_borehole_profile.pdf"}


def test_evaluate_metadata_extraction(sample_file_prediction: FilePredictions):
    """Test evaluate_metadata_extraction method of OverallFilePredictions."""
    overall_predictions = OverallFilePredictions()
    overall_predictions.add_file_predictions(sample_file_prediction)
    overall_predictions.matching_pred_to_gt_boreholes = {"example_borehole_profile.pdf": {0: 0}}

    ground_truth = GroundTruth(Path("example/example_groundtruth.json"))
    metadata_metrics = overall_predictions.evaluate_metadata_extraction(ground_truth)

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
