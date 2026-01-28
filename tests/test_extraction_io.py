"""Tests for the main extraction pipeline."""

from pathlib import Path

import numpy as np
import pytest

from extraction.main import start_pipeline

PREDICTION_FILE_ = "predictions.json"
METADATA_FILE_ = "metadata.json"
ANALYTICS_FILE_ = "matching_params_analytics.json"


@pytest.fixture
def borehole_pdf() -> Path:
    """Provide borehole file for pipeline testing.

    Returns:
        Path: Path to example file.
    """
    path_ = Path("example/example_borehole_profile.pdf")
    assert path_.exists()
    return path_


def test_start_pipeline_json(tmp_path: Path, borehole_pdf: Path) -> None:
    """Test that start_pipeline with custom prediction paths.

    Args:
        tmp_path (Path): Path to temporary folder (pytest handled).
        borehole_pdf (Path): Path to borehole PDF file.
    """
    predictions_path = tmp_path / PREDICTION_FILE_
    metadata_path = tmp_path / METADATA_FILE_

    start_pipeline(
        input_directory=borehole_pdf,
        ground_truth_path=None,
        out_directory=tmp_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        skip_draw_predictions=True,
    )

    # Check both output files exist
    assert predictions_path.exists()
    assert metadata_path.exists()


def test_start_pipeline_analytics(tmp_path: Path, borehole_pdf: Path) -> None:
    """Test that analytics are generated.

    Args:
        tmp_path (Path): Path to temporary folder (pytest handled).
        borehole_pdf (Path): Path to borehole PDF file.
    """
    start_pipeline(
        input_directory=borehole_pdf,
        ground_truth_path=None,
        out_directory=tmp_path,
        predictions_path=tmp_path / PREDICTION_FILE_,
        metadata_path=tmp_path / METADATA_FILE_,
        skip_draw_predictions=True,
        matching_analytics=True,
    )
    assert (tmp_path / ANALYTICS_FILE_).exists()


def test_start_pipeline_csv(tmp_path: Path, borehole_pdf: Path) -> None:
    """Test that CSV are generated.

    Args:
        tmp_path (Path): Path to temporary folder (pytest handled).
        borehole_pdf (Path): Path to borehole PDF file.
    """
    start_pipeline(
        input_directory=borehole_pdf,
        ground_truth_path=None,
        out_directory=tmp_path,
        predictions_path=tmp_path / PREDICTION_FILE_,
        metadata_path=tmp_path / METADATA_FILE_,
        skip_draw_predictions=True,
        csv=True,
    )
    # Generated files
    files = [f for f in tmp_path.rglob("*.csv")]
    assert len(files) != 0


def test_start_pipeline_drawing(tmp_path: Path, borehole_pdf: Path) -> None:
    """Test that CSV are generated.

    Args:
        tmp_path (Path): Path to temporary folder (pytest handled).
        borehole_pdf (Path): Path to borehole PDF file.
    """
    start_pipeline(
        input_directory=borehole_pdf,
        ground_truth_path=None,
        out_directory=tmp_path,
        predictions_path=tmp_path / PREDICTION_FILE_,
        metadata_path=tmp_path / METADATA_FILE_,
        skip_draw_predictions=False,
        draw_lines=True,
        draw_tables=True,
        draw_strip_logs=True,
    )

    # Generated files
    n_outputs = len([f for f in tmp_path.rglob("*outputs.png")])
    n_lines = len([f for f in tmp_path.rglob("*lines.png")])
    n_tables = len([f for f in tmp_path.rglob("*tables.png")])
    n_striplogs = len([f for f in tmp_path.rglob("*strip_logs.png")])
    files_lengths = np.array([n_outputs, n_lines, n_tables, n_striplogs])

    # At least one prediction per type and all equals in length
    assert np.all(files_lengths > 0)
    assert len(set(files_lengths)) == 1


def test_start_pipeline_part(tmp_path: Path, borehole_pdf: Path) -> None:
    """Test that pipeline generation is tied to part.

    Args:
        tmp_path (Path): Path to temporary folder (pytest handled).
        borehole_pdf (Path): Path to borehole PDF file.
    """
    predictions_path = tmp_path / PREDICTION_FILE_

    def infer_part(part: str = "all") -> None:
        """Inference of pipeline with part as parameter.

        Args:
            part (str, optional): Part of pipeline to run. Defaults to "all".
        """
        start_pipeline(
            input_directory=borehole_pdf,
            ground_truth_path=None,
            out_directory=tmp_path,
            predictions_path=predictions_path,
            metadata_path=tmp_path / METADATA_FILE_,
            skip_draw_predictions=True,
            part=part,
        )

    # Check inference of all vs not all
    infer_part(part="notall")
    assert not predictions_path.exists()

    infer_part(part="all")
    assert predictions_path.exists()
