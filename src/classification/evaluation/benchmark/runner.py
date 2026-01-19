"""Orchestrate running multiple classification benchmarks and aggregate results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from swissgeol_doc_processing.utils.file_utils import flatten

from .spec import BenchmarkSpec


def _flatten_metrics(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested dict into {"metrics/global_macro_f1": 0.63, ...}.

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
            if isinstance(v, (int | float)):
                out[key] = float(v)
            elif isinstance(v, str):
                try:
                    out[key] = float(v)
                except ValueError:
                    continue
    return out


def start_multi_benchmark(
    benchmarks: Sequence[BenchmarkSpec],
    out_directory: Path,
    classifier_type: str,
    model_path: Path | None,
    classification_system: str,
    out_directory_bedrock: Path,
    mlflow_tracking: bool = False,
    classification_params: dict | None = None,
):
    """Run multiple classification benchmarks in one execution.

    Output is namespaced per benchmark under:
      <out_directory>/multi/<benchmark_name>/
    """
    from classification.main import main as start_classification_pipeline  # import here to avoid cycles

    multi_root = out_directory / "multi"
    multi_root.mkdir(parents=True, exist_ok=True)

    classification_params = classification_params or {}

    parent_active = False
    if mlflow_tracking:
        import mlflow

        mlflow.set_experiment("Layer descriptions classification")
        mlflow.start_run(run_name="multi-benchmark")
        parent_active = True
        mlflow.set_tag("run_type", "multi_benchmark")
        mlflow.set_tag("benchmarks", ",".join([b.name for b in benchmarks]))
        mlflow.set_tag("classifier_type", classifier_type)
        mlflow.set_tag("classification_system", classification_system)
        if model_path:
            mlflow.set_tag("model_path", str(model_path))

        # log params once (global)
        if classification_params:
            mlflow.log_params(flatten(classification_params))

    overall_results: list[dict[str, Any]] = []

    for spec in benchmarks:
        bench_out = multi_root / spec.name
        bench_out.mkdir(parents=True, exist_ok=True)

        # Keep bedrock outputs separate per benchmark as well
        bench_out_bedrock = bench_out / "bedrock"
        bench_out_bedrock.mkdir(parents=True, exist_ok=True)

        if mlflow_tracking:
            import mlflow

            mlflow.start_run(run_name=spec.name, nested=True)
            mlflow.set_tag("benchmark_name", spec.name)
            mlflow.set_tag("file_path", str(spec.file_path))
            mlflow.set_tag(
                "file_subset_directory", str(spec.file_subset_directory) if spec.file_subset_directory else ""
            )
            mlflow.set_tag("ground_truth_path", str(spec.ground_truth_path) if spec.ground_truth_path else "")

        # Run the existing classification pipeline once
        summary = start_classification_pipeline(
            file_path=spec.file_path,
            ground_truth_path=spec.ground_truth_path,
            out_directory=bench_out,
            out_directory_bedrock=bench_out_bedrock,
            file_subset_directory=spec.file_subset_directory,
            classifier_type=classifier_type,
            model_path=model_path,
            classification_system=classification_system,
            mlflow_setup=False,  # <-- we will own the run lifecycle in multi-benchmark mode
        )

        overall_results.append({"benchmark": spec.name, "summary": summary})

        # Write per-benchmark summary artifact
        bench_summary_path = bench_out / "benchmark_summary.json"
        with open(bench_summary_path, "w", encoding="utf8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if mlflow_tracking:
            import mlflow

            mlflow.log_artifact(str(bench_summary_path), artifact_path="summary")

            # log metrics (flattened)
            if isinstance(summary, dict):
                flat = _flatten_metrics(summary, prefix="")
                if flat:
                    mlflow.log_metrics(flat)

            mlflow.end_run()

    # overall summary JSON
    summary_path = multi_root / "overall_summary.json"
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)

    # overall summary CSV (one row per benchmark)
    rows: list[dict[str, Any]] = []
    for r in overall_results:
        s = r.get("summary") or {}
        metrics = s.get("metrics") if isinstance(s, dict) else {}
        row = {
            "benchmark": r.get("benchmark"),
            "n_layers": s.get("n_layers") if isinstance(s, dict) else None,
            "file_path": s.get("file_path") if isinstance(s, dict) else None,
            "subset_dir": s.get("file_subset_directory") if isinstance(s, dict) else None,
        }
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                row[f"metrics__{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(by="benchmark")
    metric_cols = [c for c in df.columns if c.startswith("metrics__")]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce").round(4)
    df.to_csv(multi_root / "overall_summary.csv", index=False)

    if mlflow_tracking and parent_active:
        import mlflow

        mlflow.log_artifact(str(summary_path), artifact_path="summary")
        mlflow.log_artifact(str(multi_root / "overall_summary.csv"), artifact_path="summary")
        mlflow.end_run()
