from pathlib import Path
from typing import Sequence, Any
import json
import pandas as pd

from .spec import BenchmarkSpec
from swissgeol_doc_processing.utils.file_utils import flatten


def _flatten_metrics(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """
    Flatten a nested metrics dict into {"geology/layer_f1": 0.63, ...} format.
    Keeps only numeric values (int/float or strings convertible to float).
    """
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, key))
        else:
            if v is None:
                continue
            if isinstance(v, (int, float)):
                out[key] = float(v)
            elif isinstance(v, str):
                try:
                    out[key] = float(v)
                except ValueError:
                    continue
    return out


def _collect_metric_keys(overall_results: list[dict]) -> tuple[list[str], list[str]]:
    """
    Determine the union of geology and metadata metric keys across all benchmarks,
    so the CSV has stable columns even when some benchmarks have language-specific keys.
    """
    geo_keys: set[str] = set()
    meta_keys: set[str] = set()

    for r in overall_results:
        s = r.get("summary") or {}
        geo = (s.get("geology") or {}) if isinstance(s, dict) else {}
        meta = (s.get("metadata") or {}) if isinstance(s, dict) else {}

        geo_keys.update(geo.keys())
        meta_keys.update(meta.keys())

    # Put the "core" ones first, then the rest sorted
    core_geo = [
        "layer_f1", "layer_precision", "layer_recall",
        "depth_interval_f1", "depth_interval_precision", "depth_interval_recall",
        "material_description_f1", "material_description_precision", "material_description_recall",
        "groundwater_f1", "groundwater_precision", "groundwater_recall",
        "groundwater_depth_f1", "groundwater_depth_precision", "groundwater_depth_recall",
    ]
    core_meta = [
        "elevation_f1", "elevation_precision", "elevation_recall",
        "coordinate_f1", "coordinate_precision", "coordinate_recall",
        "borehole_name_f1", "borehole_name_precision", "borehole_name_recall",
    ]

    ordered_geo = [k for k in core_geo if k in geo_keys] + \
        sorted([k for k in geo_keys if k not in core_geo])
    ordered_meta = [k for k in core_meta if k in meta_keys] + \
        sorted([k for k in meta_keys if k not in core_meta])

    return ordered_geo, ordered_meta


def _make_overall_summary_rows(overall_results: list[dict]) -> list[dict[str, Any]]:
    geo_cols, meta_cols = _collect_metric_keys(overall_results)
    rows: list[dict[str, Any]] = []

    for r in overall_results:
        benchmark = r.get("benchmark")
        summary = r.get("summary") or {}

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
    """
    Run multiple benchmarks in one execution.

    Output is namespaced per benchmark under:
      <out_directory>/multi/<benchmark_name>/
    """
    from extraction.main import start_pipeline
    multi_root = out_directory / "multi"
    multi_root.mkdir(parents=True, exist_ok=True)
    line_detection_params = line_detection_params or {}
    matching_params = matching_params or {}
    # Parent MLflow run (optional) + nested runs per benchmark
    parent_active = False
    if mlflow_tracking == True:
        import mlflow
        mlflow.set_experiment("Boreholes data extraction")
        mlflow.start_run(run_name="multi-benchmark")
        parent_active = True
        mlflow.set_tag("run_type", "multi_benchmark")
        mlflow.set_tag("benchmarks", ",".join([b.name for b in benchmarks]))
        # log params once (they're global)
        mlflow.log_params(flatten(line_detection_params))
        mlflow.log_params(flatten(matching_params))

    overall_results = []

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

        overall_results.append(
            {"benchmark": spec.name, "summary": eval_result})

        if mlflow_tracking:
            import mlflow
            # If eval_result is a dict of metrics, log them
            if isinstance(eval_result, dict):
                flat_metrics = _flatten_metrics(eval_result)
                if flat_metrics:
                    mlflow.log_metrics(flat_metrics)

                # Also log benchmark summary json as an artifact inside the child run
                bench_summary_path = bench_out / "benchmark_summary.json"
                with open(bench_summary_path, "w", encoding="utf8") as f:
                    json.dump(eval_result, f, ensure_ascii=False, indent=2)
                mlflow.log_artifact(str(bench_summary_path),
                                    artifact_path="summary")

            mlflow.end_run()

    # Log an overall summary artifact + optional aggregate metrics
    summary_path = multi_root / "overall_summary.json"
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)

    # Write a readable CSV comparison table
    summary_csv_path = multi_root / "overall_summary.csv"
    rows = _make_overall_summary_rows(overall_results)

    df = pd.DataFrame(rows)

    # sort benchmarks alphabetically (or by a key metric)
    df = df.sort_values(by="benchmark")

    # Add aggregate rows (means across benchmarks) ---
    metric_cols = [c for c in df.columns if c.startswith("geology__") or c.startswith("metadata__")]

    df_metrics = df.copy()
    df_metrics[metric_cols] = df_metrics[metric_cols].apply(pd.to_numeric, errors="coerce")
    df_metrics["n_documents"] = pd.to_numeric(df_metrics["n_documents"], errors="coerce")

    # Unweighted mean (simple average across benchmarks)
    unweighted_means = {}
    for c in metric_cols:
        col = df_metrics[c]
        unweighted_means[c] = float(col.mean()) if col.notna().any() else None

    mean_unweighted_row = {
        "benchmark": "MEAN_unweighted",
        "ground_truth_path": "",
        "n_documents": int(df_metrics["n_documents"].sum()) if df_metrics["n_documents"].notna().any() else "",
        **unweighted_means,
    }

    # Weighted mean by n_documents
    weighted_means = {}
    weights = df_metrics["n_documents"].fillna(0)

    for c in metric_cols:
        vals = df_metrics[c]
        mask = vals.notna() & (weights > 0)
        if mask.any():
            weighted_means[c] = float((vals[mask] * weights[mask]).sum() / weights[mask].sum())
        else:
            weighted_means[c] = None

    mean_weighted_row = {
        "benchmark": "MEAN_weighted_by_n_documents",
        "ground_truth_path": "",
        "n_documents": int(weights.sum()),
        **weighted_means,
    }

    # Append both rows
    df = pd.concat([df, pd.DataFrame([mean_unweighted_row, mean_weighted_row])], ignore_index=True)

    # keep 3 decimals for readability
    numeric_cols = [c for c in df.columns if c.startswith(
        "geology__") or c.startswith("metadata__")]
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce").round(3)

    df.to_csv(summary_csv_path, index=False)

    if mlflow_tracking and parent_active:
        import mlflow
        mlflow.log_artifact(str(summary_path), artifact_path="summary")
        mlflow.log_artifact(str(summary_csv_path), artifact_path="summary")
        mlflow.end_run()
