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

    overlap_result = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert overlap_result is None


def test_find_last_duplicate_single_at_top(create_layer):
    """Test when there is a single duplicate layer at the bottom."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B")]
    current_layers = [create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    overlap_result = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert overlap_result.upper_id == 2
    assert overlap_result.lower_id == 1


def test_find_last_duplicate_multiple_consecutive(create_layer):
    """Test when there are multiple consecutive duplicate layers."""
    prev_layers = [create_layer("Layer A"), create_layer("Layer B"), create_layer("Layer C")]
    current_layers = [create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    overlap_result = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert overlap_result.upper_id == 3
    assert overlap_result.lower_id == 2


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
    overlap_result = find_split_by_convolution(prev_layers, current_layers_correct, matching_params)
    assert overlap_result.upper_id == 4
    assert overlap_result.lower_id == 2

    current_layers_correct = [
        create_layer("Layer X"),
        create_layer("Layer Y"),
        create_layer("Layer D"),  # false positive, same description but not a duplicate
        create_layer("Layer Z"),
    ]
    overlap_result = find_split_by_convolution(prev_layers, current_layers_correct, matching_params)
    assert overlap_result is None


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

    overlap_result = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert overlap_result.upper_id == 2
    assert overlap_result.lower_id == 1


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

    overlap_result = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert overlap_result.upper_id == 2
    assert overlap_result.lower_id == 1


def test_find_last_duplicate_ocr_error(create_layer):
    """Test when layers have partial text matches that should be considered duplicates."""
    prev_layers = [create_layer("Clay with gravel and some sand"), create_layer("Sand, silt with organic material")]
    current_layers = [
        create_layer("Sand. silt with organic material"),  # ocr error: . instead of , (reason of using Levenshtein)
        create_layer("Clay with gravel"),
    ]

    overlap_result = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert overlap_result.upper_id == 2
    assert overlap_result.lower_id == 1


def test_find_last_duplicate_full_duplicates(create_layer):
    """Test when the current page has one new leading layer before the full duplicated block.

    The plain window search can't align "Layer A" against anything in prev, but dropping it reveals
    that the rest of the page (B, C, D) is a full duplicate of prev.
    """
    prev_layers = [create_layer("Layer B"), create_layer("Layer C"), create_layer("Layer D")]

    current_layers_correct = [
        create_layer("Layer A"),
        create_layer("Layer B"),
        create_layer("Layer C"),
        create_layer("Layer D"),
    ]
    overlap_result = find_split_by_convolution(prev_layers, current_layers_correct, matching_params)
    assert overlap_result.upper_id == 3
    assert overlap_result.lower_id == 4


def test_find_last_duplicate_unmatched_boundary_layer(create_elevation_layer):
    """A boundary layer that doesn't compare well to its counterpart must not block detection.

    Regression test inspired by 3384.pdf: the current page's first layer's text was OCR-truncated
    enough that it no longer matched its counterpart in prev, which used to prevent the sliding
    window from ever aligning with the real duplicate content right after it.
    """
    prev_layers = [
        create_elevation_layer("dito grau", (26.1, 27.4)),
        create_elevation_layer("Lehm mit Sandstein", (27.4, 27.95)),
        create_elevation_layer("Mergel grau, fest, braun bis rotlich", (27.95, 28.75)),
        create_elevation_layer("Sandstein zerbrockelt", (28.75, 29.95)),
    ]
    current_layers = [
        create_elevation_layer("fest braun bis rotlich", (None, 28.75)),  # truncated, doesn't match prev's layer
        create_elevation_layer("Sandstein zerbrockelt", (28.75, 29.95)),
        create_elevation_layer("Mergel bunt", (29.95, 31.6)),
    ]

    overlap_result = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert overlap_result is not None
    assert overlap_result.upper_id == 4
    assert overlap_result.lower_id == 2


def test_find_split_by_convolution_depth_reset_overlap(create_elevation_layer):
    """Test depth reset overlap.

    A ruler blocking the OCR text at the page boundary breaks text matching, but the current page's
    depths restart lower while reproducing several exact depth values from the previous page - this
    must still be detected as an overlap of the re-scanned section, not a new borehole.
    """
    prev_layers = [
        create_elevation_layer("Sandstein grau", (11.3, 11.55)),
        create_elevation_layer("Sandstein hart", (11.55, 12.6)),
        create_elevation_layer("Mergel bunt", (12.6, 13.6)),
        create_elevation_layer("Mergel bunt Fortsetzung", (13.6, 16.15)),
    ]
    current_layers = [
        create_elevation_layer("XXXX unlesbar unter Lineal", (11.3, 11.55)),
        create_elevation_layer("XXXX unlesbar unter Lineal", (11.55, 12.6)),
        create_elevation_layer("XXXX unlesbar unter Lineal", (12.6, 13.6)),
        create_elevation_layer("XXXX unlesbar unter Lineal", (13.6, 16.15)),
        create_elevation_layer("Sandstein neu", (16.15, 18.1)),
    ]

    overlap_result = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert overlap_result is not None
    assert overlap_result.upper_id == 4
    assert overlap_result.lower_id == 4


def test_find_split_by_convolution_depth_reset_no_overlap(create_elevation_layer):
    """A current page that genuinely starts a new, shallower borehole must not be merged in.

    Only one depth value coincides with the previous borehole (12.6) - not enough evidence of a
    real re-scanned overlap.
    """
    prev_layers = [
        create_elevation_layer("Sandstein grau", (11.3, 11.55)),
        create_elevation_layer("Sandstein hart", (11.55, 12.6)),
    ]
    current_layers = [
        create_elevation_layer("Humus", (0.0, 0.5)),
        create_elevation_layer("Lehm", (0.5, 3.0)),
        create_elevation_layer("Kies", (3.0, 12.6)),
    ]

    overlap_result = find_split_by_convolution(prev_layers, current_layers, matching_params)
    assert overlap_result is None


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
    assert not are_layers_similar(layer_a, layer_b)
    assert are_layers_similar(layer_a, layer_c)


def test_are_layers_similar_extremities(create_layer):
    """Test if two layers are matched based on position in file."""
    layer_a = create_layer("Desc A Desc B")
    layer_b = create_layer("Desc A")
    layer_c = create_layer("Desc B")
    assert not are_layers_similar(layer_a, layer_b, is_extremity=False)
    assert not are_layers_similar(layer_a, layer_b, is_extremity=True)
    assert are_layers_similar(layer_a, layer_c, is_extremity=True)
    assert not are_layers_similar(layer_c, layer_a, is_extremity=True)
