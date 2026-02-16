"""Tests for the main extraction pipeline."""

import os
from io import BytesIO
from pathlib import Path

import pymupdf
import pytest

# Enforce MLFlow tracking to False before importing modules
os.environ["MLFLOW_TRACKING"] = "False"

from extraction.evaluation.benchmark.spec import BenchmarkSpec
from extraction.main import start_pipeline
from extraction.runner import extract, start_pipeline_benchmark

PREDICTION_FILE_ = "predictions.json"
METADATA_FILE_ = "metadata.json"
ANALYTICS_FILE_ = "matching_params_analytics.json"
OVERALL_SUMMARY = "overall_summary.csv"


@pytest.fixture
def borehole_pdf() -> Path:
    """Provide borehole file for pipeline testing.

    Returns:
        Path: Path to example file.
    """
    path_ = Path("example/example_borehole_profile.pdf")
    assert path_.exists()
    return path_


@pytest.fixture
def borehole_gt() -> Path:
    """Provide borehole file for pipeline testing.

    Returns:
        Path: Path to example file.
    """
    path_ = Path("example/example_groundtruth.json")
    assert path_.exists()
    return path_


def test_file_stream_extract(borehole_pdf: Path) -> None:
    """Test that core borehole function work with stream.

    Args:
        borehole_pdf (Path): Path to borehole PDF file.
    """
    # Test extract from stream
    prediction_stream = extract(
        file=BytesIO(pymupdf.Document(borehole_pdf).tobytes()),
        filename=borehole_pdf.name,
    )

    # Test extract from path
    prediction_path = extract(
        file=borehole_pdf,
        filename=borehole_pdf.name,
    )

    assert prediction_stream is not None
    assert prediction_path is not None


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

    # Check that temporary files are cleaned
    assert len([f for f in predictions_path.parent.rglob("*.tmp")]) == 0


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
    # Check generated csv files
    assert len([f for f in tmp_path.rglob("*.csv")]) != 0


def test_start_pipeline_drawing(tmp_path: Path, borehole_pdf: Path) -> None:
    """Test that visualizations are generated.

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
    file_types = ["outputs", "lines", "tables", "strip_logs"]
    file_counts = [len([f for f in tmp_path.rglob(f"*{file_type}.png")]) for file_type in file_types]

    # At least one prediction per type and all equals in length
    assert all(count > 0 for count in file_counts)
    assert len(set(file_counts)) == 1


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
            part (str): Pipeline mode, "all" for full extraction, "metadata" for metadata only. Defaults to "all".
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
    infer_part(part="metadata")
    assert not predictions_path.exists()

    infer_part(part="all")
    assert predictions_path.exists()


def test_start_pipeline_nested(tmp_path: Path, borehole_pdf: Path) -> None:
    """Test that pipeline resumes from temporary file and skips already processed files.

    Args:
        tmp_path (Path): Path to temporary folder (pytest handled).
        borehole_pdf (Path): Path to borehole PDF file.
    """
    predictions_path = tmp_path / PREDICTION_FILE_
    metadata_path = tmp_path / METADATA_FILE_
    predictions_path_tmp = tmp_path / (PREDICTION_FILE_ + ".tmp")

    # Run first time
    start_pipeline(
        input_directory=borehole_pdf,
        ground_truth_path=None,
        out_directory=tmp_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        skip_draw_predictions=True,
        is_nested=True,
    )

    # Verify tmp exists
    assert predictions_path_tmp.exists()

    # Run again - should skip already processed
    start_pipeline(
        input_directory=borehole_pdf,
        ground_truth_path=None,
        out_directory=tmp_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        skip_draw_predictions=True,
        is_nested=False,
    )

    # Verify tmp was removed
    assert not predictions_path_tmp.exists()


def test_start_pipeline_benchmark(tmp_path: Path, borehole_gt: Path, borehole_pdf: Path) -> None:
    """Test that pipeline benchmark generation create files.

    Args:
        tmp_path (Path): Path to temporary folder (pytest handled).
        borehole_gt (Path): Path to ground truth file.
        borehole_pdf (Path): Path to borehole PDF file.
    """
    # Parse specs
    specs = [
        BenchmarkSpec(name="bench_1", input_path=borehole_pdf, ground_truth_path=borehole_gt),
        BenchmarkSpec(name="bench_2", input_path=borehole_pdf, ground_truth_path=borehole_gt),
    ]

    # Run main pipeline
    start_pipeline_benchmark(
        benchmarks=specs,
        out_directory=tmp_path,
        skip_draw_predictions=True,
    )

    # Checkout outputs for all benchmarks
    for spec in specs:
        # Check main folder
        assert (tmp_path / spec.name).exists()
        # Check predictions and meta data
        assert (tmp_path / spec.name / PREDICTION_FILE_).exists()
        assert (tmp_path / spec.name / METADATA_FILE_).exists()

    # Check aggregation
    assert (tmp_path / OVERALL_SUMMARY).exists()

    # Check that temporary files are cleaned
    assert len([f for f in tmp_path.rglob("*.tmp")]) == 0
