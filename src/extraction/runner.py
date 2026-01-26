"""Orchestrate running multiple benchmarks and aggregate results."""

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import pymupdf
from tqdm import tqdm

from extraction import DATAPATH
from extraction.annotations.draw import draw_predictions, plot_strip_logs, plot_tables
from extraction.annotations.plot_utils import plot_lines, save_visualization
from extraction.evaluation.benchmark.score import evaluate_all_predictions
from extraction.features.extract import extract_page
from extraction.features.groundwater.groundwater_extraction import (
    GroundwaterInDocument,
    GroundwaterLevelExtractor,
)
from extraction.features.metadata.borehole_name_extraction import NameInDocument, extract_borehole_names
from extraction.features.metadata.metadata import FileMetadata, MetadataInDocument
from extraction.features.predictions.borehole_predictions import BoreholePredictions
from extraction.features.predictions.file_predictions import FilePredictions
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from extraction.features.predictions.predictions import BoreholeListBuilder
from extraction.features.stratigraphy.layer.continuation_detection import merge_boreholes
from extraction.features.stratigraphy.layer.layer import LayersInDocument
from swissgeol_doc_processing.geometry.line_detection import extract_lines
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.text.matching_params_analytics import create_analytics
from swissgeol_doc_processing.utils.file_utils import flatten, read_params
from swissgeol_doc_processing.utils.strip_log_detection import detect_strip_logs
from swissgeol_doc_processing.utils.table_detection import detect_table_structures
from utils.benchmark_utils import _short_metric_key, _shorten_metric_dict

from .evaluation.benchmark.spec import BenchmarkSpec

mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"  # Checks whether MLFlow tracking is enabled
if mlflow_tracking:
    import mlflow
    import pygit2

matching_params = read_params("matching_params.yml")
line_detection_params = read_params("line_detection_params.yml")
name_detection_params = read_params("name_detection_params.yml")
table_detection_params = read_params("table_detection_params.yml")
striplog_detection_params = read_params("striplog_detection_params.yml")

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _flatten_metrics(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten a nested metrics dict into {"geology/layer_f1": 0.63, ...} format.

    Keeps only numeric values (int/float or strings convertible to float).

    Args:
        d (dict[str, Any]): Nested dict of metrics.
        prefix (str, optional): Prefix for keys (used in recursion). Defaults to "".

    Returns:
        dict[str, float]: Flattened dict of metrics.
    """
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"

        if v is None:
            continue
        elif isinstance(v, dict):
            out.update(_flatten_metrics(v, key))
        elif isinstance(v, (int | float)):
            out[key] = float(v)
        elif isinstance(v, str):
            try:
                out[key] = float(v)
            except ValueError:
                continue
    return out


def _collect_metric_keys(overall_results: list[dict[str, Any]]) -> list[str]:
    """Union of flattened metric keys across all benchmarks.

    We treat any numeric leaf in the summary dict as a metric.
    Non-dict summaries are skipped defensively.
    """
    keys: set[str] = set()

    for result in overall_results:
        summary = result.get("summary")
        if not isinstance(summary, dict):
            continue

        flat = _flatten_metrics(summary)

        keys.update(flat.keys())

    return sorted(keys)


def _make_overall_summary_rows(overall_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create rows for the overall summary CSV from benchmark results (generic metrics)."""
    metric_keys = _collect_metric_keys(overall_results)
    rows: list[dict[str, Any]] = []

    for result in overall_results:
        benchmark = result.get("benchmark")
        summary = result.get("summary")

        base: dict[str, Any] = {
            "benchmark": benchmark,
            "ground_truth_path": summary.get("ground_truth_path") if isinstance(summary, dict) else None,
            "n_documents": summary.get("n_documents") if isinstance(summary, dict) else None,
        }

        flat = _flatten_metrics(summary) if isinstance(summary, dict) else {}

        # Put metrics as columns; use __ instead of / for CSV friendliness
        for k in metric_keys:
            base[k.replace("/", "__")] = flat.get(k)

        rows.append(base)

    return rows


def _setup_mlflow_parent_run(
    *,
    mlflow_tracking: bool,
    benchmarks: Sequence[BenchmarkSpec],
    line_detection_params: dict,
    matching_params: dict,
) -> bool:
    """Start the parent MLflow run (multi-benchmark) and log global params once.

    Args:
        mlflow_tracking (bool): Whether MLflow tracking is enabled.
        benchmarks (Sequence[BenchmarkSpec]): List of benchmark specs.
        line_detection_params (dict): Line detection parameters to log.
        matching_params (dict): Matching parameters to log.

    Returns:
        bool: True if a parent run was started and must be closed by the caller.
    """
    if not mlflow_tracking:
        return False

    import mlflow

    mlflow.set_experiment("Boreholes data extraction")

    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.start_run()
    mlflow.set_tag("run_type", "multi_benchmark")
    mlflow.set_tag("benchmarks", ",".join([b.name for b in benchmarks]))

    mlflow.log_params(flatten(line_detection_params))
    mlflow.log_params(flatten(matching_params))
    return True


def _finalize_overall_summary(
    *,
    overall_results: list[dict[str, Any]],
    multi_root: Path,
    mlflow_tracking: bool,
    parent_active: bool,
) -> tuple[Path, Path]:
    """Write overall_summary.json and overall_summary.csv (+ mean row).

    Also logs overall aggregate metrics + artifacts to MLflow on the parent run (if enabled).
    """
    # --- JSON ---
    summary_path = multi_root / "overall_summary.json"
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)

    # --- CSV ---
    summary_csv_path = multi_root / "overall_summary.csv"
    rows = _make_overall_summary_rows(overall_results)
    df = pd.DataFrame(rows).sort_values(by="benchmark")

    # Everything except these base columns is a metric column
    base_cols = {"benchmark", "ground_truth_path", "n_documents"}
    metric_cols = [c for c in df.columns if c not in base_cols]

    df_metrics = df.copy()
    df_metrics[metric_cols] = df_metrics[metric_cols].apply(pd.to_numeric, errors="coerce")
    df_metrics["n_documents"] = pd.to_numeric(df_metrics["n_documents"], errors="coerce")

    means: dict[str, float | None] = {}
    for c in metric_cols:
        col = df_metrics[c]
        means[c] = float(col.mean()) if col.notna().any() else None

    mean_row = {
        "benchmark": "mean",
        "ground_truth_path": "",
        "n_documents": int(df_metrics["n_documents"].sum()) if df_metrics["n_documents"].notna().any() else "",
        **means,
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce").round(3)
    df.to_csv(summary_csv_path, index=False)

    # --- MLflow: overall mean metrics + artifacts on parent run ---
    if mlflow_tracking and parent_active:
        import mlflow

        overall_mean_metrics: dict[str, float] = {}
        for k, v in means.items():
            if v is None:
                continue

            full_key = k.replace("__", "/")  # only needed if your CSV columns use "__"
            short_key = _short_metric_key(full_key)

            # collision-safe:
            key_to_log = short_key if short_key not in overall_mean_metrics else full_key
            overall_mean_metrics[key_to_log] = float(v)

        mlflow.log_metrics(overall_mean_metrics)

        total_docs = df_metrics["n_documents"].sum(skipna=True)
        if pd.notna(total_docs):
            mlflow.log_metric("total_n_documents", float(total_docs))

        mlflow.log_artifact(str(summary_path), artifact_path="summary")
        mlflow.log_artifact(str(summary_csv_path), artifact_path="summary")


def setup_mlflow_tracking(
    input_directory: Path,
    ground_truth_path: Path,
    out_directory: Path = None,
    predictions_path: Path = None,
    metadata_path: Path = None,
    experiment_name: str = "Boreholes data extraction",
):
    """Set up MLFlow tracking."""
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tag("input_directory", str(input_directory))
    mlflow.set_tag("ground_truth_path", str(ground_truth_path))
    if out_directory:
        mlflow.set_tag("out_directory", str(out_directory))
    if predictions_path:
        mlflow.set_tag("predictions_path", str(predictions_path))
    if metadata_path:
        mlflow.set_tag("metadata_path", str(metadata_path))
    mlflow.log_params(flatten(line_detection_params))
    mlflow.log_params(flatten(matching_params))

    repo = pygit2.Repository(".")
    commit = repo[repo.head.target]
    mlflow.set_tag("git_branch", repo.head.shorthand)
    mlflow.set_tag("git_commit_message", commit.message)
    mlflow.set_tag("git_commit_sha", commit.id)


def start_pipeline(
    input_directory: Path,
    ground_truth_path: Path,
    out_directory: Path,
    predictions_path: Path,
    metadata_path: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
    csv: bool = False,
    matching_analytics: bool = False,
    part: str = "all",
    mlflow_setup: bool = True,
    temp_directory: Path | None = None,
):
    """Run the boreholes data extraction pipeline.

    The pipeline will extract material description of all found layers and assign them to the corresponding
    depth intervals. The input directory should contain pdf files with boreholes data. The algorithm can deal
    with borehole profiles of multiple pages.

    Note: This function is used to be called from the label-studio backend, whereas the click_pipeline function
    is called from the CLI.

    Args:
        input_directory (Path): The directory containing the pdf files. Can also be the path to a single pdf file.
        ground_truth_path (Path | None): The path to the ground truth file json file.
        out_directory (Path): The directory to store the evaluation results.
        predictions_path (Path): The path to the predictions file.
        skip_draw_predictions (bool, optional): Whether to skip drawing predictions on pdf pages. Defaults to False.
        draw_lines (bool, optional): Whether to draw lines on pdf pages. Defaults to False.
        draw_tables (bool, optional): Whether to draw detected table structures on pdf pages. Defaults to False.
        draw_strip_logs (bool, optional): Whether to draw detected strip log structures on pages. Defaults to False.
        metadata_path (Path): The path to the metadata file.
        csv (bool): Whether to generate a CSV output. Defaults to False.
        matching_analytics (bool): Whether to enable matching parameters analytics. Defaults to False.
        part (str, optional): The part of the pipeline to run. Defaults to "all".
        mlflow_setup (bool, optional): Whether to set up MLFlow tracking. Defaults to True.
        temp_directory (Path | None, optional): Temporary directory for intermediate files. Defaults to None.
    """  # noqa: D301
    # Initialize analytics if enabled
    analytics = create_analytics() if matching_analytics else None

    if mlflow_tracking and mlflow_setup:
        setup_mlflow_tracking(input_directory, ground_truth_path, out_directory, predictions_path, metadata_path)
    # temporary directory to dump files for mlflow artifact logging / evaluation artifacts
    if temp_directory is None:
        temp_directory = DATAPATH / "_temp"
    temp_directory.mkdir(parents=True, exist_ok=True)

    if skip_draw_predictions:
        draw_directory = None
    else:
        # check if directories exist and create them when necessary
        draw_directory = out_directory / "draw"
        draw_directory.mkdir(parents=True, exist_ok=True)

    if csv:
        csv_dir = out_directory / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

    # if a file is specified instead of an input directory, copy the file to a temporary directory and work with that.
    if input_directory.is_file():
        root = input_directory.parent
        files = [input_directory.name]
    else:
        root = input_directory
        _, _, files = next(os.walk(input_directory))

    # process the individual pdf files
    predictions = OverallFilePredictions()

    for filename in tqdm(files, desc="Processing files", unit="file"):
        if not filename.endswith(".pdf"):
            logger.warning(f"{filename} does not end with .pdf and is not treated.")
            continue

        in_path = os.path.join(root, filename)
        logger.info("Processing file: %s", in_path)

        with pymupdf.Document(in_path) as doc:
            # Extract metadata
            file_metadata = FileMetadata.from_document(doc, matching_params)
            metadata = MetadataInDocument.from_document(doc, file_metadata.language, matching_params)

            # Save the predictions to the overall predictions object, initialize common variables
            all_groundwater_entries = GroundwaterInDocument([], filename)
            all_name_entries = NameInDocument([], filename)
            boreholes_per_page = []

            if part != "all":
                continue
            # Extract the layers
            for page_index, page in enumerate(doc):
                page_number = page_index + 1
                logger.info("Processing page %s", page_number)

                text_lines = extract_text_lines(page)
                long_or_horizontal_lines, all_geometric_lines = extract_lines(page, line_detection_params)
                name_entries = extract_borehole_names(text_lines, name_detection_params)
                all_name_entries.name_feature_list.extend(name_entries)

                # Detect table structures on the page
                table_structures = detect_table_structures(
                    page_index, doc, long_or_horizontal_lines, text_lines, table_detection_params
                )

                # Detect strip logs on the page
                strip_logs = detect_strip_logs(page, text_lines, striplog_detection_params)

                # extract the statigraphy
                page_layers = extract_page(
                    text_lines,
                    long_or_horizontal_lines,
                    all_geometric_lines,
                    table_structures,
                    strip_logs,
                    file_metadata.language,
                    page_index,
                    doc,
                    line_detection_params,
                    analytics,
                    **matching_params,
                )
                boreholes_per_page.append(page_layers)

                # Extract the groundwater levels
                groundwater_extractor = GroundwaterLevelExtractor(file_metadata.language, matching_params)
                groundwater_entries = groundwater_extractor.extract_groundwater(
                    page_number=page_number,
                    text_lines=text_lines,
                    geometric_lines=long_or_horizontal_lines,
                    extracted_boreholes=page_layers,
                )
                all_groundwater_entries.groundwater_feature_list.extend(groundwater_entries)

                # Draw table structures if requested
                if draw_tables:
                    img = plot_tables(page, table_structures, page_index)
                    save_visualization(img, filename, page.number + 1, "tables", draw_directory, mlflow_tracking)

                # Draw strip logs if requested
                if draw_strip_logs:
                    img = plot_strip_logs(page, strip_logs, page_index)
                    save_visualization(img, filename, page.number + 1, "strip_logs", draw_directory, mlflow_tracking)

                if draw_lines:  # could be changed to if draw_lines and mlflow_tracking:
                    if not mlflow_tracking:
                        logger.warning("MLFlow tracking is not enabled. MLFLow is required to store the images.")
                    else:
                        img = plot_lines(
                            page,
                            all_geometric_lines,
                            scale_factor=line_detection_params["pdf_scale_factor"],
                        )
                        mlflow.log_image(img, f"pages/{filename}_page_{page.number + 1}_lines.png")

            layers_with_bb_in_document = LayersInDocument(
                merge_boreholes(boreholes_per_page, matching_params), filename
            )

            # create list of BoreholePrediction objects with all the separate lists
            borehole_predictions_list: list[BoreholePredictions] = BoreholeListBuilder(
                layers_with_bb_in_document=layers_with_bb_in_document,
                file_name=filename,
                groundwater_in_doc=all_groundwater_entries,
                names_in_doc=all_name_entries,
                elevations_list=metadata.elevations,
                coordinates_list=metadata.coordinates,
            ).build()
            # now that the matching is done, duplicated groundwater can be removed and depths info can be set
            for borehole in borehole_predictions_list:
                borehole.filter_groundwater_entries()

            # Add file predictions
            predictions.add_file_predictions(FilePredictions(borehole_predictions_list, file_metadata, filename))

            # Add layers to a csv file
            if csv:
                base_path = csv_dir / Path(filename).stem

                for index, borehole in enumerate(borehole_predictions_list):
                    csv_path = f"{base_path}_{index}.csv" if len(borehole_predictions_list) > 1 else f"{base_path}.csv"
                    logger.info("Writing CSV predictions to %s", csv_path)
                    with open(csv_path, "w", encoding="utf8", newline="") as file:
                        file.write(borehole.to_csv())

                    if mlflow_tracking:
                        mlflow.log_artifact(csv_path, "csv")

    logger.info("Metadata written to %s", metadata_path)
    with open(metadata_path, "w", encoding="utf8") as file:
        json.dump(predictions.get_metadata_as_dict(), file, ensure_ascii=False)

    if part == "all":
        logger.info("Writing predictions to JSON file %s", predictions_path)
        with open(predictions_path, "w", encoding="utf8") as file:
            json.dump(predictions.to_json(), file, ensure_ascii=False)

    eval_summary = evaluate_all_predictions(
        predictions=predictions,
        ground_truth_path=ground_truth_path,
        temp_directory=temp_directory,
        mlflow_tracking=mlflow_tracking,
    )

    if input_directory and draw_directory:
        draw_predictions(predictions, input_directory, draw_directory)

    # Finalize analytics if enabled
    if matching_analytics:
        analytics_output_path = out_directory / "matching_params_analytics.json"
        analytics.save_analytics(analytics_output_path)
        logger.info(f"Matching parameters analytics saved to {analytics_output_path}")

    return eval_summary


def start_pipeline_benchmark(
    benchmarks: Sequence[BenchmarkSpec],
    out_directory: Path,
    skip_draw_predictions: bool = False,
    draw_lines: bool = False,
    draw_tables: bool = False,
    draw_strip_logs: bool = False,
    csv: bool = False,
    matching_analytics: bool = False,
    part: str = "all",
    mlflow_tracking: bool = False,
    line_detection_params: dict | None = None,
    matching_params: dict | None = None,
):
    """Run multiple benchmarks in one execution.

    Args:
        benchmarks (Sequence[BenchmarkSpec]): List of benchmark specifications.
        out_directory (Path): Output directory for multi-benchmark results.
        skip_draw_predictions (bool, optional): Whether to skip drawing predictions. Defaults to False.
        draw_lines (bool, optional): Whether to draw detected lines. Defaults to False.
        draw_tables (bool, optional): Whether to draw detected tables. Defaults to False.
        draw_strip_logs (bool, optional): Whether to draw strip logs. Defaults to False.
        csv (bool, optional): Whether to output CSV summaries. Defaults to False.
        matching_analytics (bool, optional): Whether to compute matching analytics. Defaults to False.
        part (str, optional): Part of the pipeline to run. Defaults to "all".
        mlflow_tracking (bool, optional): Whether to enable MLflow tracking. Defaults to False.
        line_detection_params (dict, optional): Line detection parameters to log. Defaults to None.
        matching_params (dict, optional): Matching parameters to log. Defaults to None.

    Output is namespaced per benchmark under:
      <out_directory>/multi/<benchmark_name>/
    """
    multi_root = out_directory / "multi"
    multi_root.mkdir(parents=True, exist_ok=True)
    line_detection_params = line_detection_params or {}
    matching_params = matching_params or {}

    parent_active = _setup_mlflow_parent_run(
        mlflow_tracking=mlflow_tracking,
        benchmarks=benchmarks,
        line_detection_params=line_detection_params,
        matching_params=matching_params,
    )

    overall_results = []
    try:
        for spec in benchmarks:
            bench_out = multi_root / spec.name
            bench_out.mkdir(parents=True, exist_ok=True)

            bench_predictions_path = bench_out / "predictions.json"
            bench_metadata_path = bench_out / "metadata.json"

            if mlflow_tracking:
                import mlflow

                mlflow.start_run(run_name=spec.name, nested=True)
                mlflow.set_tag("benchmark_name", spec.name)
                mlflow.set_tag("input_directory", str(spec.input_path))
                mlflow.set_tag("ground_truth_path", str(spec.ground_truth_path))

            with TemporaryDirectory(prefix=f"{spec.name}_", dir=str(bench_out)) as td:
                bench_temp = Path(td)

            eval_result = start_pipeline(
                input_directory=spec.input_path,
                ground_truth_path=spec.ground_truth_path,
                out_directory=bench_out,
                predictions_path=bench_predictions_path,
                metadata_path=bench_metadata_path,
                skip_draw_predictions=skip_draw_predictions,
                draw_lines=draw_lines,
                draw_tables=draw_tables,
                draw_strip_logs=draw_strip_logs,
                csv=csv,
                matching_analytics=matching_analytics,
                part=part,
                mlflow_setup=False,
                temp_directory=bench_temp,
            )

            overall_results.append({"benchmark": spec.name, "summary": eval_result})

            if mlflow_tracking:
                import mlflow

                if isinstance(eval_result, dict):
                    flat_metrics = _flatten_metrics(eval_result)
                    short_metrics = _shorten_metric_dict(flat_metrics)

                    if short_metrics:
                        mlflow.log_metrics(short_metrics)

                    bench_summary_path = bench_out / "benchmark_summary.json"
                    with open(bench_summary_path, "w", encoding="utf8") as f:
                        json.dump(eval_result, f, ensure_ascii=False, indent=2)
                    mlflow.log_artifact(str(bench_summary_path), artifact_path="summary")

                mlflow.end_run()

        _finalize_overall_summary(
            overall_results=overall_results,
            multi_root=multi_root,
            mlflow_tracking=mlflow_tracking,
            parent_active=parent_active,
        )

    finally:
        if mlflow_tracking and parent_active:
            import mlflow

            if mlflow.active_run() is not None:
                mlflow.end_run()
