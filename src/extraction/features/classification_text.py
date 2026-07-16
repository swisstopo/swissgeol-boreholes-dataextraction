"""Per-borehole document text extraction, for use as input to document-level classifiers (e.g. borehole_type).

Each borehole's text is limited to the pages where that borehole was actually detected (via its
`bounding_boxes`), so that a single PDF containing multiple boreholes yields distinct text per borehole.
"""

from __future__ import annotations

import dataclasses
from io import BytesIO
from pathlib import Path

from extraction.core.extract import extract, open_pdf
from extraction.features.predictions.borehole_predictions import BoreholePredictions
from swissgeol_doc_processing.text.extract_text import extract_text_lines, filter_header_candidate_lines
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.file_utils import read_params

matching_params = read_params("matching_params.yml")


@dataclasses.dataclass
class BoreholeText:
    """The text associated with a single borehole detected in a file."""

    borehole_index: int
    text: str


def borehole_text_key(filename: str, borehole_index: int) -> str:
    """Build the compound key used to store/look up a single borehole's text in a filename->text JSON mapping."""
    return f"{filename}::{borehole_index}"


def _borehole_pages(borehole: BoreholePredictions) -> list[int]:
    """Return the sorted, deduplicated 1-indexed page numbers a borehole's bounding boxes span."""
    return sorted({bboxes.page for bboxes in borehole.bounding_boxes})


def extract_borehole_texts(file: Path | BytesIO, filename: str, header_only: bool = False) -> list[BoreholeText]:
    """Extract per-borehole text from a PDF, scoped to the pages each borehole was detected on.

    Args:
        file (Path | BytesIO): Path or stream of the PDF file to process.
        filename (str): Name of the file used as identifier.
        header_only (bool): If True, drop material description lines and standalone numbers (e.g. depths),
            keeping only the header-like remainder of the text. Defaults to exporting the full text.

    Returns:
        list[BoreholeText]: One entry per borehole detected in the file, in `borehole_index` order.
    """
    result = extract(file=file, filename=filename, part="all")
    language = result.predictions.file_metadata.language

    with open_pdf(file=file, filename=filename) as doc:
        borehole_texts = []
        for borehole in result.predictions.borehole_predictions_list:
            lines: list[TextLine] = [
                line for page_number in _borehole_pages(borehole) for line in extract_text_lines(doc[page_number - 1])
            ]
            if header_only:
                lines = filter_header_candidate_lines(lines, language, matching_params)

            borehole_texts.append(
                BoreholeText(
                    borehole_index=borehole.borehole_index,
                    text="\n".join(line.text for line in lines),
                )
            )

    return borehole_texts
