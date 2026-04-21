"""Test suite for overlap detection."""

import pymupdf
import pytest

from extraction.features.stratigraphy.layer.layer import Layer, LayerDepths, LayerDepthsEntry
from extraction.features.stratigraphy.layer.overlap_detection import are_layers_similar, find_split_by_convolution
from swissgeol_doc_processing.text.textblock import MaterialDescription, MaterialDescriptionLine
from swissgeol_doc_processing.utils.data_extractor import FeatureOnPage
from swissgeol_doc_processing.utils.file_utils import read_params

matching_params = read_params("matching_params.yml")


@pytest.fixture
def create_layer():
    """Create a Layer with given text."""

    def _create_layer(text: str) -> Layer:
        line_feat = FeatureOnPage(MaterialDescriptionLine(text), rect=pymupdf.Rect, page=0)
        material_description = MaterialDescription(text, [line_feat])
        return Layer(material_description=material_description, depths=None)

    return _create_layer


@pytest.fixture
def create_elevation_layer():
    """Create a Layer with given text and depth interval."""

    def _create_elevation_layer(text: str, elevation: tuple[int | None, int | None]) -> Layer:
        line_feat = FeatureOnPage(MaterialDescriptionLine(text), rect=pymupdf.Rect, page=0)
        material_description = MaterialDescription(text, [line_feat])
        depths = LayerDepths(
            LayerDepthsEntry(elevation[0], rect=pymupdf.Rect, page_number=0),
            LayerDepthsEntry(elevation[1], rect=pymupdf.Rect, page_number=0),
        )
        return Layer(material_description=material_description, depths=depths)

    return _create_elevation_layer


def test_find_last_duplicate_no_duplicates(create_layer):
    """Test when there are no duplicate layers."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B")]
    current_layers = [create_layer("Layer C"), create_layer("Layer D")]

    bottom_duplicated_idx = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert bottom_duplicated_idx is None


def test_find_last_duplicate_single_at_top(create_layer):
    """Test when there is a single duplicate layer at the bottom."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B")]
    current_layers = [create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    bottom_duplicated_idx = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert bottom_duplicated_idx == 1


def test_find_last_duplicate_multiple_consecutive(create_layer):
    """Test when there are multiple consecutive duplicate layers."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B"), create_layer("Layer C")]
    current_layers = [create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    bottom_duplicated_idx = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert bottom_duplicated_idx == 2


def test_find_last_duplicate_false_positive(create_layer):
    """Test when a layer has the same description, but is not a duplicate (above layer do not match)."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    current_layers_correct = [
        create_layer("Layer C"),
        create_layer("Layer D"),  # real duplicates
        create_layer("Layer Y"),
        create_layer("Layer D"),  # false positive, same description but not a duplicate
        create_layer("Layer Z"),
    ]
    bottom_duplicated_idx = find_split_by_convolution(prev_layers, current_layers_correct, matching_params)
    assert bottom_duplicated_idx == 2

    current_layers_correct = [
        create_layer("Layer X"),
        create_layer("Layer Y"),
        create_layer("Layer D"),  # false positive, same description but not a duplicate
        create_layer("Layer Z"),
    ]
    bottom_duplicated_idx = find_split_by_convolution(prev_layers, current_layers_correct, matching_params)
    assert bottom_duplicated_idx is None


def test_find_last_duplicate_prev_bottom_cropped(create_layer):
    """Test when the bottom of the description on the previous page is cropped."""
    prev_layers = [
        create_layer("Dirt"),
        create_layer("Silty soil"),  # with rocks cropped
    ]
    current_layers = [
        create_layer("Silty soil with rocks"),
        create_layer("Bedrock"),
    ]

    bottom_duplicated_idx = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert bottom_duplicated_idx == 1


def test_find_last_duplicate_current_top_cropped(create_layer):
    """Test when the top of the description on the current page is cropped."""
    prev_layers = [
        create_layer("Dirt"),
        create_layer("Silty soil with rocks"),
    ]
    current_layers = [
        create_layer("with rocks"),  # Silty soil cropped
        create_layer("Bedrock"),
    ]

    bottom_duplicated_idx = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert bottom_duplicated_idx == 1


def test_find_last_duplicate_ocr_error(create_layer):
    """Test when layers have partial text matches that should be considered duplicates."""
    prev_layers = [create_layer("Clay with gravel and some sand"), create_layer("Sand, silt with organic material")]
    current_layers = [
        create_layer("Sand. silt with organic material"),  # ocr error: . instead of , (reason of using Levenshtein)
        create_layer("Clay with gravel"),
    ]

    bottom_duplicated_idx = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert bottom_duplicated_idx == 1


def test_find_last_duplicate_full_duplicates(create_layer):
    """Test when the number of duplicated layers of the current page exceeds the number of layer in the previous."""
    prev_layers = [create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    current_layers_correct = [
        create_layer("Layer A"),
        create_layer("Layer B"),
        create_layer("Layer C"),
        create_layer("Layer D"),
    ]
    bottom_duplicated_idx = find_split_by_convolution(prev_layers, current_layers_correct, matching_params)
    assert bottom_duplicated_idx is None


def test_are_layers_similar_text(create_layer):
    """Test if two layers are matched based on small text difference."""
    layer_a = create_layer("Desc A")
    layer_b = create_layer("Desc AB")
    assert are_layers_similar(layer_a, layer_b, material_threshold=0.80)
    assert not are_layers_similar(layer_a, layer_b, material_threshold=1.00)


def test_are_layers_similar_elevation(create_elevation_layer):
    """Test if two layers are matched based on elevation difference."""
    layer_a = create_elevation_layer("Desc A", [0, 1])
    layer_b = create_elevation_layer("Desc A", [0, 2])
    layer_c = create_elevation_layer("Desc A", [0, None])
    assert are_layers_similar(layer_a, layer_b, force_depth_matching=False)
    assert not are_layers_similar(layer_a, layer_b, force_depth_matching=True)
    assert are_layers_similar(layer_a, layer_c, force_depth_matching=True)


def test_are_layers_similar_extremities(create_layer):
    """Test if two layers are matched based on position in file."""
    layer_a = create_layer("Desc A Desc B")
    layer_b = create_layer("Desc A")
    layer_c = create_layer("Desc B")
    assert not are_layers_similar(layer_a, layer_b, is_extremity=False)
    assert not are_layers_similar(layer_a, layer_b, is_extremity=True)
    assert are_layers_similar(layer_a, layer_c, is_extremity=True)
    assert not are_layers_similar(layer_c, layer_a, is_extremity=True)
