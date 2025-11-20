"""Test module for striplog detection and merging."""

import pymupdf
import pytest

from swissgeol_doc_processing.utils.strip_log_detection import StripLogSection, _is_ocr_artifact, _rescale_bboxes


@pytest.mark.parametrize(
    "bbox_source, bbox_candidate, r_tol, aligns",
    [
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([100, 20, 200, 30]), 0, True),
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([100, 0, 200, 10]), 0, True),
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([50, 20, 150, 30]), 0.5, True),
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([150, 0, 250, 10]), 0.5, True),
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([49, 0, 149, 10]), 0.5, False),
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([151, 0, 251, 10]), 0.5, False),
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([100, 20, 250, 30]), 0.5, True),
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([100, 0, 150, 10]), 0.5, True),
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([100, 10, 251, 20]), 0.5, False),
        (pymupdf.Rect([100, 10, 200, 20]), pymupdf.Rect([100, 10, 149, 20]), 0.5, False),
    ],
    ids=["above", "below", "left", "right", "too-left", "too-right", "wide", "short", "too-wide", "too-short"],
)
def test_striplog_section_aligns(
    bbox_source: pymupdf.Rect, bbox_candidate: pymupdf.Rect, r_tol: float, aligns: bool
) -> None:
    """Validate vertical alignment logic between two strip-log sections.

    Cases cover candidates directly above/below the source and horizontally
    shifted/width-mismatched variants. Alignment is considered valid when the
    candidate lies vertically adjacent to the source and its horizontal span
    overlaps within a relative tolerance `r_tol`.

    Args:
        bbox_source (pymupdf.Rect): Bounding box of the source section.
        bbox_candidate (pymupdf.Rect): Bounding box of the candidate section to compare against.
        r_tol (float): Relative horizontal tolerance.
        aligns (bool): Expected boolean outcome for the given pair and tolerance.
    """
    source = StripLogSection(bbox=bbox_source)
    candidate = StripLogSection(bbox=bbox_candidate)
    assert source.aligns(candidate, r_tol=r_tol) == aligns


@pytest.mark.parametrize(
    "bbox, scale, bbox_rescaled",
    [
        (pymupdf.Rect([0, 0, 100, 10]), 1, pymupdf.Rect([0, 0, 100, 10])),
        (pymupdf.Rect([0, 0, 100, 10]), 0.5, pymupdf.Rect([0, 0, 50, 5])),
        (pymupdf.Rect([0, 0, 100, 10]), 2, pymupdf.Rect([0, 0, 200, 20])),
    ],
    ids=["same", "shrink", "expand"],
)
def test_rescale_bboxes(bbox: pymupdf.Rect, scale: float, bbox_rescaled: pymupdf.Rect) -> None:
    """Ensure bounding boxes are uniformly rescaled by a scalar factor.

    Args:
        bbox (pymupdf.Rect): Input rectangle to scale.
        scale (float): Uniform scale factor applied.
        bbox_rescaled (pymupdf.Rect): Expected rectangle after scaling.
    """
    assert _rescale_bboxes([bbox], scale)[0] == bbox_rescaled


@pytest.mark.parametrize(
    "text, expected",
    [
        ("018", True),
        ("2345679", False),
        ("oO", True),
        ("ab", False),
        (" ", True),
        ("", False),
        ("|/()-.,=_", True),
        ("%@&", False),
        ("0.( ) 88 //- 1", True),
        ("numeric zero 0.1", False),
    ],
    ids=[
        "digits-artifact",
        "digits-non-artifact",
        "letters-artifact",
        "letters-non-artifact",
        "whitespace-only",
        "empty",
        "punctuation-artifact",
        "punctuation-non-artifact",
        "text-artifact",
        "text-non-artifact",
    ],
)
def test_is_ocr_artifact(text: str, expected: bool) -> None:
    """Check detection of 'numeric-like filler' patterns.

    Args:
        text (str): Input string to classify.
        expected (bool): Expected boolean classification.
    """
    assert _is_ocr_artifact(text) == expected
