"""Orchestrate running multiple benchmarks and aggregate results."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from swissgeol_doc_processing.utils.file_utils import flatten

from .spec import BenchmarkSpec


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


def _collect_metric_keys(overall_results: list[dict]) -> tuple[list[str], list[str]]:
    """Determine the union of geology and metadata metric keys across all benchmarks.

    This way, the CSV has stable columns even when some benchmarks have language-specific keys.

    Args:
        overall_results (list[dict]): List of benchmark results.

    Returns:
        tuple[list[str], list[str]]: (geology metric keys, metadata metric keys)
    """
    geo_keys: set[str] = set()
    meta_keys: set[str] = set()

    for result in overall_results:
        summary = result.get("summary")
        if not isinstance(summary, dict):
            continue

        geo = summary.get("geology")
        if isinstance(geo, dict):
            geo_keys.update(geo.keys())

        meta = summary.get("metadata")
        if isinstance(meta, dict):
            meta_keys.update(meta.keys())

    return sorted(geo_keys), sorted(meta_keys)


def _make_overall_summary_rows(overall_results: list[dict]) -> list[dict[str, Any]]:
    """Create rows for the overall summary CSV from benchmark results.

    Args:
        overall_results (list[dict]): List of benchmark results.

    Returns:
        list[dict[str, Any]]: List of rows for the CSV.
    """
    geo_cols, meta_cols = _collect_metric_keys(overall_results)
    rows: list[dict[str, Any]] = []

    for result in overall_results:
        benchmark = result.get("benchmark")
        summary = result.get("summary") or {}

        base = {
            "benchmark": benchmark,
            "ground_truth_path": summary.get("ground_truth_path") if isinstance(summary, dict) else None,
            "n_documents": summary.get("n_documents") if isinstance(summary, dict) else None,
        }

        geo = summary.get("geology", {}) if isinstance(summary, dict) else {}
        meta = summary.get("metadata", {}) if isinstance(summary, dict) else {}

        # Prefix to avoid name collisions and to group columns nicely
        for k in geo_cols:
            base[f"geology__{k}"] = geo.get(k)
        for k in meta_cols:
            base[f"metadata__{k}"] = meta.get(k)

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

    Args:
        overall_results (list[dict[str, Any]]): List of benchmark results.
        multi_root (Path): Path to the multi-benchmark output directory.
        mlflow_tracking (bool): Whether MLflow tracking is enabled.
        parent_active (bool): Whether a parent MLflow run is active.

    Returns:
        tuple[Path, Path]: (summary_json_path, summary_csv_path)
    """
    # --- JSON ---
    summary_path = multi_root / "overall_summary.json"
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)

    # --- CSV (per-benchmark + MEAN row) ---
    summary_csv_path = multi_root / "overall_summary.csv"
    rows = _make_overall_summary_rows(overall_results)
    df = pd.DataFrame(rows).sort_values(by="benchmark")

    metric_cols = [c for c in df.columns if c.startswith("geology__") or c.startswith("metadata__")]

    df_metrics = df.copy()
    df_metrics[metric_cols] = df_metrics[metric_cols].apply(pd.to_numeric, errors="coerce")
    df_metrics["n_documents"] = pd.to_numeric(df_metrics["n_documents"], errors="coerce")

    means: dict[str, float | None] = {}
    for c in metric_cols:
        col = df_metrics[c]
        means[c] = float(col.mean()) if col.notna().any() else None

    mean_row = {
        "benchmark": "MEAN",
        "ground_truth_path": "",
        "n_documents": int(df_metrics["n_documents"].sum()) if df_metrics["n_documents"].notna().any() else "",
        **means,
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    numeric_cols = [c for c in df.columns if c.startswith("geology__") or c.startswith("metadata__")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").round(3)

    df.to_csv(summary_csv_path, index=False)

    # --- MLflow: overall mean metrics + artifacts on parent run ---
    if mlflow_tracking and parent_active:
        import mlflow

        overall_mean_metrics: dict[str, float] = {}
        for k, v in means.items():
            if v is None:
                continue
            overall_key = "overall_mean/" + k.replace("__", "/")
            overall_mean_metrics[overall_key] = float(v)

        if overall_mean_metrics:
            mlflow.log_metrics(overall_mean_metrics)

        total_docs = df_metrics["n_documents"].sum(skipna=True)
        if pd.notna(total_docs):
            mlflow.log_metric("overall/total_n_documents", float(total_docs))

        mlflow.log_artifact(str(summary_path), artifact_path="summary")
        mlflow.log_artifact(str(summary_csv_path), artifact_path="summary")

    return summary_path, summary_csv_path


def start_multi_benchmark(
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
    #  import here to avoid circular imports
    from extraction.main import start_pipeline

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

            bench_temp = bench_out / "_temp"
            bench_temp.mkdir(parents=True, exist_ok=True)

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
                    if flat_metrics:
                        mlflow.log_metrics(flat_metrics)

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
