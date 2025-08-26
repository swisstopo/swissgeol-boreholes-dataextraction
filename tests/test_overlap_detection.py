"""Test suite for overlap detection."""

import pymupdf
import pytest

from extraction.features.stratigraphy.layer.layer import Layer
from extraction.features.stratigraphy.layer.overlap_detection import find_last_duplicate_layer_index
from extraction.features.utils.data_extractor import FeatureOnPage
from extraction.features.utils.text.textblock import MaterialDescription, MaterialDescriptionLine


@pytest.fixture
def create_layer():
    """Create a Layer with given text."""

    def _create_layer(text: str) -> Layer:
        line_feat = FeatureOnPage(MaterialDescriptionLine(text), rect=pymupdf.Rect, page=0)
        material_description = MaterialDescription(text, [line_feat])
        return Layer(material_description=material_description, depths=None)

    return _create_layer


def test_find_last_duplicate_no_duplicates(create_layer):
    """Test when there are no duplicate layers."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B")]
    current_layers = [create_layer("Layer C"), create_layer("Layer D")]

    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers)
    assert bottom_duplicated_idx is None


def test_find_last_duplicate_single_at_top(create_layer):
    """Test when there is a single duplicate layer at the bottom."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B")]
    current_layers = [create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers)
    assert bottom_duplicated_idx == 0


def test_find_last_duplicate_multiple_consecutive(create_layer):
    """Test when there are multiple consecutive duplicate layers."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B"), create_layer("Layer C")]
    current_layers = [create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers)
    assert bottom_duplicated_idx == 1


def test_find_last_duplicate_wrong_line_grouping(create_layer):
    """Test when description lines are grouped differently on each page, due to different scan quality."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B"), create_layer("Layer C")]
    current_layers = [
        create_layer("Layer B"),
        create_layer("Layer"),  # description incorrectly split,
        create_layer("C"),  # the algorithm will also pick up this layer
        create_layer("Layer D"),
    ]

    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers)
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
    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers_correct)
    assert bottom_duplicated_idx == 1

    current_layers_correct = [
        create_layer("Layer X"),
        create_layer("Layer Y"),
        create_layer("Layer D"),  # false positive, same description but not a duplicate
        create_layer("Layer Z"),
    ]
    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers_correct)
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

    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers)
    assert bottom_duplicated_idx == 0


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

    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers)
    assert bottom_duplicated_idx == 0


def test_find_last_duplicate_ocr_error(create_layer):
    """Test when layers have partial text matches that should be considered duplicates."""
    prev_layers = [create_layer("Clay with gravel and some sand"), create_layer("Sand, silt with organic material")]
    current_layers = [
        create_layer("Sand. silt with organic material"),  # ocr error: . instead of , (reason of using Levenshtein)
        create_layer("Clay with gravel"),
    ]

    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers)
    assert bottom_duplicated_idx == 0


def test_find_last_duplicate_full_duplicates(create_layer):
    """Test when the number of duplicated layer of the current page exeeds the number of layer in the previous."""
    prev_layers = [create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    current_layers_correct = [
        create_layer("Layer A"),
        create_layer("Layer B"),
        create_layer("Layer C"),
        create_layer("Layer D"),
    ]
    bottom_duplicated_idx = find_last_duplicate_layer_index(prev_layers, current_layers_correct)
    assert bottom_duplicated_idx == 3
