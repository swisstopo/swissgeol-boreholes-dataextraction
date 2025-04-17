"""Test suite for the prediction module."""

from datetime import datetime
from unittest.mock import Mock

import pymupdf
import pytest
from borehole_extraction.evaluation.benchmark.ground_truth import GroundTruth
from borehole_extraction.evaluation.layer_evaluator import LayerEvaluator
from borehole_extraction.evaluation.utility import evaluate, evaluate_single
from borehole_extraction.extraction.groundwater.groundwater_extraction import Groundwater, GroundwatersInBorehole
from borehole_extraction.extraction.metadata.coordinate_extraction import CoordinateEntry, LV95Coordinate
from borehole_extraction.extraction.metadata.metadata import BoreholeMetadata, FileMetadata
from borehole_extraction.extraction.predictions.borehole_predictions import (
    BoreholePredictions,
    BoreholePredictionsWithGroundTruth,
    FilePredictionsWithGroundTruth,
)
from borehole_extraction.extraction.predictions.file_predictions import FilePredictions
from borehole_extraction.extraction.predictions.overall_file_predictions import OverallFilePredictions
from borehole_extraction.extraction.predictions.predictions import AllBoreholePredictionsWithGroundTruth
from borehole_extraction.extraction.stratigraphy.layer.layer import (
    Layer,
    LayerDepths,
    LayerDepthsEntry,
    LayersInBorehole,
)
from borehole_extraction.extraction.util_extraction.data_extractor.data_extractor import FeatureOnPage
from borehole_extraction.extraction.util_extraction.text.textblock import MaterialDescription


@pytest.fixture
def sample_file_prediction() -> FilePredictions:
    """Fixture to create a sample FilePredictions object."""
    filename = "example_borehole_profile.pdf"
    coord = FeatureOnPage(
        feature=LV95Coordinate(
            east=CoordinateEntry(coordinate_value=2789456), north=CoordinateEntry(coordinate_value=1123012)
        ),
        rect=pymupdf.Rect(),
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
        rect=pymupdf.Rect(0, 0, 100, 100),
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


@pytest.fixture
def file_prediction_with_two_boreholes() -> FilePredictions:
    """Fixture to create a sample FilePredictions object that has two boreholes."""
    filename = "example_borehole_profile.pdf"
    coord = FeatureOnPage(
        feature=LV95Coordinate(
            east=CoordinateEntry(coordinate_value=2789456), north=CoordinateEntry(coordinate_value=1123012)
        ),
        rect=pymupdf.Rect(),
        page=1,
    )

    layers_in_borehole = LayersInBorehole(
        [
            Layer(
                material_description=FeatureOnPage(MaterialDescription(text=descr, lines=[]), pymupdf.Rect(), 0),
                depths=LayerDepths(LayerDepthsEntry(start, pymupdf.Rect()), LayerDepthsEntry(end, pymupdf.Rect())),
            )
            for descr, start, end in [
                ("HUMUS", None, 1),
                ("KIES, grau", 1, 2),
                ("sand", 2, 3),
            ]
        ]
    )
    layers_in_borehole_2 = LayersInBorehole(
        [
            Layer(
                material_description=FeatureOnPage(MaterialDescription(text=descr, lines=[]), pymupdf.Rect(), 0),
                depths=LayerDepths(LayerDepthsEntry(start, pymupdf.Rect()), LayerDepthsEntry(end, pymupdf.Rect())),
            )
            for descr, start, end in [
                ("KIES, Sand,", 0.0, 0.5),
                ("stein, sand", 0.5, 2.0),
            ]
        ]
    )

    dt_date = datetime(2024, 10, 1)
    groundwater_on_page = FeatureOnPage(
        feature=Groundwater(depth=100, date=dt_date, elevation=20),
        page=1,
        rect=pymupdf.Rect(0, 0, 100, 100),
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
            ),
            BoreholePredictions(
                borehole_index=1,
                layers_in_borehole=layers_in_borehole_2,
                file_name=filename,
                metadata=metadata,
                groundwater_in_borehole=groundwater_in_bh,
                bounding_boxes=[],
            ),
        ],
        file_name=filename,
        file_metadata=file_metadata,
    )


@pytest.fixture
def groundtruth():
    """Path to the ground truth file."""
    return GroundTruth("example/example_groundtruth.json")


@pytest.fixture
def groundtruth_with_two_boreholes():
    """Path to the ground truth file that has two boreholes."""
    return GroundTruth("example/example_layers_groundtruth.json")


@pytest.fixture
def sample_file_prediction_with_ground_truth(
    sample_file_prediction: FilePredictions, groundtruth: GroundTruth
) -> FilePredictionsWithGroundTruth:
    """Builds a FilePredictionsWithGroundTruth object with the given predictions and groundtruth.

    Args:
        sample_file_prediction (FilePredictions): a fixture that returns the FilePredictions object
        groundtruth (GroundTruth): a fixture that returns the Groudtruth object

    Returns:
        FilePredictionsWithGroundTruth: the FilePredictionsWithGroundTruth associated
    """
    file_ground_truth: dict = groundtruth.for_file(sample_file_prediction.file_name)
    gt_index = 0
    return FilePredictionsWithGroundTruth(
        filename=sample_file_prediction.file_name,
        language=sample_file_prediction.file_metadata.language,
        boreholes=[
            BoreholePredictionsWithGroundTruth(
                predictions=borehole_preds, ground_truth=file_ground_truth.get(gt_index)
            )
            for borehole_preds in sample_file_prediction.borehole_predictions_list
        ],
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


def test_evaluate_layer_matching(
    file_prediction_with_two_boreholes: FilePredictions, groundtruth_with_two_boreholes: GroundTruth
):
    """Test the matching of predictions to ground truths when multiple boreholes are present in one document."""
    groundtruth_for_file = groundtruth_with_two_boreholes.for_file("example_borehole_profile.pdf")
    sample_file_prediction_with_ground_truth: FilePredictionsWithGroundTruth = (
        LayerEvaluator.match_predictions_with_ground_truth(file_prediction_with_two_boreholes, groundtruth_for_file)
    )
    # We test the matching by comparing the number of layers, one borehole has 2, the other has 3.
    assert all(
        [
            len(pred.predictions.layers_in_borehole.layers) == len(pred.ground_truth["layers"])
            for pred in sample_file_prediction_with_ground_truth
        ]
    )


def test_evaluate_metadata_extraction(sample_file_prediction_with_ground_truth: FilePredictionsWithGroundTruth):
    """Test evaluate_metadata_extraction method of OverallFilePredictions."""
    all_predictions_with_gt = AllBoreholePredictionsWithGroundTruth([sample_file_prediction_with_ground_truth])
    metadata_metrics = all_predictions_with_gt.evaluate_metadata_extraction()

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
def test_evaluate(values, ground_truth, expected):
    """Test count_against_ground_truth with various scenarios."""
    metrics = evaluate(values, ground_truth, lambda a, b: a == b).metrics
    assert (metrics.tp, metrics.fp, metrics.fn) == expected


@pytest.mark.parametrize(
    "value,ground_truth,expected",
    [
        (1, 1, (1, 0, 0)),
        (1, 2, (0, 1, 1)),
        (1, None, (0, 1, 0)),
        (None, 1, (0, 0, 1)),
        (None, None, (0, 0, 0)),
    ],
)
def test_evaluate_single(value, ground_truth, expected):
    """Test count_against_ground_truth with various scenarios."""
    metrics = evaluate_single(value, ground_truth, lambda a, b: a == b).metrics
    assert (metrics.tp, metrics.fp, metrics.fn) == expected
