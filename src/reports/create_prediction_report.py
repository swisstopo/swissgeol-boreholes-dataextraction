"""Build a W&B report comparing a prediction run against a baseline run.

Uses `wandb.Api()` to validate the two runs, read metadata for the report text, and pull
each run's logged `png_browser_table` to build a side-by-side image comparison table.
The side-by-side table is logged onto whichever run is already active, which the caller owns.
"""

import logging

import wandb_workspaces.expr as expr
import wandb_workspaces.reports.v2 as wr

import wandb
from reports.comparison_table import build_comparison_table, build_file_metric_comparison_tables, load_table

logger = logging.getLogger(__name__)

COMPARISON_TABLE_KEY = "prediction_comparison_table"
PNG_BROWSER_TABLE_KEY = "png_browser_table"
FILE_METRIC_COMPARISON_TABLE_KEY = "prediction_file_metric_comparison_table"
FILE_METRIC_CHANGES_TABLE_KEY = "prediction_file_metric_changes_table"

FILE_METRIC_COLUMNS = [
    "name_f1",
    "name_recall",
    "name_precision",
    "coordinates_f1",
    "coordinates_recall",
    "coordinates_precision",
    "elevation_f1",
    "elevation_recall",
    "elevation_precision",
    "layer_f1",
    "layer_recall",
    "layer_precision",
    "layer_num_total",
    "layer_num_wrong",
    "material_description_f1",
    "material_description_recall",
    "material_description_precision",
    "depth_interval_f1",
    "depth_interval_recall",
    "depth_interval_precision",
    "groundwater_f1",
    "groundwater_recall",
    "groundwater_precision",
    "groundwater_depth_f1",
    "groundwater_depth_recall",
    "groundwater_depth_precision",
    "groundwater_depth_num_detected",
]


def _log_comparison_table(prediction_run, baseline_run, prediction_run_id: str, baseline_run_id: str) -> bool:
    """Build the baseline-vs-prediction side-by-side image table and log it onto the active run.

    Returns:
        bool: True if the table was built and logged, False if it was skipped.
    """
    if wandb.run is None or wandb.run.id != prediction_run_id:
        logger.warning(
            "create_prediction_report must be called from within the active prediction run; "
            "skipping side-by-side comparison table."
        )
        return False

    try:
        rows_pred, dir_pred = load_table(prediction_run, PNG_BROWSER_TABLE_KEY)
        rows_base, dir_base = load_table(baseline_run, PNG_BROWSER_TABLE_KEY)
    except Exception:
        logger.exception(
            "Could not load %r from prediction/baseline runs; skipping comparison table. "
            "Note: runs predating per-row image table logging (no dedicated 'png_browser_table' "
            "artifact, only a bare history artifact) can never provide this - pick a newer baseline run.",
            PNG_BROWSER_TABLE_KEY,
        )
        return False

    table, gallery, *_ = build_comparison_table(
        rows_base,
        dir_base,
        rows_pred,
        dir_pred,
        join_key="filename",
        image_column="image",
        run_a_id=baseline_run_id,
        run_b_id=prediction_run_id,
    )
    wandb.log({COMPARISON_TABLE_KEY: table, "prediction_comparison_gallery": gallery})
    return True


def _log_file_metric_comparison_tables(
    prediction_run, baseline_run, prediction_run_id: str, baseline_run_id: str
) -> bool:
    """Build the baseline-vs-prediction file-level metric comparison tables and log them onto the active run.

    Unlike `_log_comparison_table` (which joins by the PNG filename to pair up images),
    this dedupes `png_browser_table`'s repeated per-PNG rows down to one row per file/document
    (by `pdf_filename` + `page`) before comparing metrics.

    Returns:
        bool: True if the tables were built and logged, False if it was skipped.
    """
    if wandb.run is None or wandb.run.id != prediction_run_id:
        logger.warning(
            "create_prediction_report must be called from within the active prediction run; "
            "skipping file-level metric comparison tables."
        )
        return False

    try:
        rows_pred, _ = load_table(prediction_run, PNG_BROWSER_TABLE_KEY)
        rows_base, _ = load_table(baseline_run, PNG_BROWSER_TABLE_KEY)
    except Exception:
        logger.exception(
            "Could not load %r from prediction/baseline runs; skipping file-level metric comparison tables.",
            PNG_BROWSER_TABLE_KEY,
        )
        return False

    wide_table, long_table, shared_keys, only_in_base, only_in_prediction = build_file_metric_comparison_tables(
        rows_base,
        rows_pred,
        metric_columns=FILE_METRIC_COLUMNS,
        run_a_id=baseline_run_id,
        run_b_id=prediction_run_id,
    )

    if only_in_base:
        logger.warning(
            "%d file(s) only in baseline %s, excluded from file-level metric comparison",
            len(only_in_base),
            baseline_run_id,
        )
    if only_in_prediction:
        logger.warning(
            "%d file(s) only in prediction %s, excluded from file-level metric comparison",
            len(only_in_prediction),
            prediction_run_id,
        )
    if shared_keys:
        conflict_base_idx = wide_table.columns.index("duplicate_metric_conflict_base")
        conflict_pred_idx = wide_table.columns.index("duplicate_metric_conflict_prediction")
        num_conflicts = sum(1 for row in wide_table.data if row[conflict_base_idx] or row[conflict_pred_idx])
        if num_conflicts:
            logger.warning(
                "%d file(s) have conflicting metric values across duplicate png_browser_table rows",
                num_conflicts,
            )

    wandb.log({FILE_METRIC_COMPARISON_TABLE_KEY: wide_table, FILE_METRIC_CHANGES_TABLE_KEY: long_table})
    return True


def create_prediction_report(
    entity: str,
    project: str,
    prediction_run_id: str,
    baseline_run_id: str,
    media_keys: list[str] = ["png_browser"],  # noqa: B006
    metric_keys: list[str] | None = None,
) -> str:
    """Create (and save) a W&B report comparing a prediction run against a baseline run.

    Args:
        entity (str): W&B entity.
        project (str): W&B project.
        prediction_run_id (str): Run id of the prediction run being reported on.
        baseline_run_id (str): Run id of the baseline run to compare against.
        media_keys (list[str]): Media log keys to render in a MediaBrowser panel (e.g. "png_browser").
        metric_keys (list[str] | None): Optional summary metric keys to plot in a BarPlot, if present
            in either run's summary.

    Returns:
        str: URL of the saved report.
    """
    api = wandb.Api()
    prediction_run = api.run(f"{entity}/{project}/{prediction_run_id}")
    baseline_run = api.run(f"{entity}/{project}/{baseline_run_id}")

    comparison_table_logged = _log_comparison_table(prediction_run, baseline_run, prediction_run_id, baseline_run_id)
    file_metric_comparison_logged = _log_file_metric_comparison_tables(
        prediction_run, baseline_run, prediction_run_id, baseline_run_id
    )

    baseline_runset = wr.Runset(
        entity=entity,
        project=project,
        name="Baseline",
        filters=expr.And(expr.Metric("name").isin([baseline_run_id])),
    )
    prediction_runset = wr.Runset(
        entity=entity,
        project=project,
        name="Prediction",
        filters=expr.And(expr.Metric("name").isin([prediction_run_id])),
    )

    title = f"Prediction comparison — {prediction_run.display_name}".replace("/", "-")

    summary_lines = [
        f"- **Prediction run:** {prediction_run.display_name} (`{prediction_run.id}`)",
        f"- **Baseline run:** {baseline_run.display_name} (`{baseline_run.id}`)",
        f"- **Created:** {prediction_run.created_at}",
        f"- [Prediction run]({prediction_run.url}) · [Baseline run]({baseline_run.url})",
    ]
    if file_metric_comparison_logged:
        summary_lines.append(
            f"- File-level metric comparison logged as `{FILE_METRIC_COMPARISON_TABLE_KEY}` "
            f"and `{FILE_METRIC_CHANGES_TABLE_KEY}`."
        )
    summary_md = "\n".join(summary_lines)

    media_keys_text = ", ".join(f"`{key}`" for key in media_keys)

    blocks = [
        wr.H1(text=title),
        wr.MarkdownBlock(text=summary_md),
        wr.MarkdownBlock(
            text=(
                "## Run comparison\n\n"
                "Side-by-side summary metrics and config values for the baseline and prediction runs."
            )
        ),
        wr.PanelGrid(
            runsets=[baseline_runset, prediction_runset],
            panels=[wr.RunComparer(layout=wr.Layout(w=24, h=12))],
        ),
        wr.MarkdownBlock(
            text=(f"## Prediction images\n\nBrowse the logged {media_keys_text} images side-by-side for both runs.")
        ),
        *[
            wr.PanelGrid(
                runsets=[baseline_runset, prediction_runset],
                panels=[wr.MediaBrowser(media_keys=[media_key], layout=wr.Layout(w=24, h=12))],
            )
            for media_key in media_keys
        ],
    ]

    if metric_keys:
        present_metrics = [k for k in metric_keys if k in prediction_run.summary or k in baseline_run.summary]
        if present_metrics:
            metrics_text = ", ".join(f"`{metric}`" for metric in present_metrics)
            blocks.append(
                wr.MarkdownBlock(
                    text=(
                        "## Summary metric comparison\n\n"
                        f"Bar chart comparing {metrics_text} between the baseline and prediction runs."
                    )
                )
            )
            blocks.append(
                wr.PanelGrid(
                    runsets=[baseline_runset, prediction_runset],
                    panels=[wr.BarPlot(metrics=present_metrics, layout=wr.Layout(w=24, h=9))],
                )
            )

    if comparison_table_logged:
        blocks.append(
            wr.MarkdownBlock(
                text=(
                    "## Side-by-side image comparison\n\n"
                    f"Table `{COMPARISON_TABLE_KEY}` has one row per file. Each run's own image sits in its "
                    f"own column, named after that run's id (`{baseline_run_id}`, `{prediction_run_id}`), so "
                    "the two render side by side in the same row.\n\n"
                    '- Tip: filter with `row["filename"] == "<filename>"`.'
                )
            )
        )
        blocks.append(
            wr.PanelGrid(
                runsets=[prediction_runset],
                panels=[wr.WeavePanelSummaryTable(table_name=COMPARISON_TABLE_KEY, layout=wr.Layout(w=24, h=16))],
            )
        )

    if file_metric_comparison_logged:
        blocks.append(
            wr.MarkdownBlock(
                text=(
                    "## File-level metric changes\n\n"
                    "The wide table has one row per file/document. Useful columns include "
                    "`pdf_filename`, `page`, `num_changed_metrics`, `mean_abs_delta`, "
                    "`max_abs_delta`, and `biggest_changed_metric`.\n\n"
                    "The long-form table has one row per file and metric. Useful columns include "
                    "`pdf_filename`, `metric`, `baseline_value`, `prediction_value`, `delta`, "
                    "`abs_delta`, `changed`, and `direction`.\n\n"
                    '- Tip: filter the wide table with `row["num_changed_metrics"] > 0` to show only '
                    "files with changes.\n"
                    '- Tip: in the long-form table, filter with `row["changed"]==True`, '
                    '`row["metric"] == "layer_f1"`, or `row["pdf_filename"] == "<file>.pdf"`.\n'
                    '- Tip: sort the long-form table by `row["abs_delta"]` descending to find the '
                    "biggest changes."
                )
            )
        )

        blocks.append(
            wr.PanelGrid(
                runsets=[prediction_runset],
                panels=[
                    wr.WeavePanelSummaryTable(
                        table_name=FILE_METRIC_COMPARISON_TABLE_KEY,
                        layout=wr.Layout(w=24, h=16),
                    )
                ],
            )
        )

        blocks.append(
            wr.PanelGrid(
                runsets=[prediction_runset],
                panels=[
                    wr.WeavePanelSummaryTable(
                        table_name=FILE_METRIC_CHANGES_TABLE_KEY,
                        layout=wr.Layout(w=24, h=16),
                    )
                ],
            )
        )

    report = wr.Report(entity=entity, project=project, title=title, width="fluid", blocks=blocks)
    report.save()

    saved = wr.Report.from_url(report.url)
    assert saved.title == title, f"Saved report title {saved.title!r} does not match expected {title!r}"
    panel_grids = [b for b in saved.blocks if isinstance(b, wr.PanelGrid)]
    assert panel_grids, "Saved report has no PanelGrid blocks"
    assert any(isinstance(p, wr.RunComparer) for grid in panel_grids for p in grid.panels), (
        "Saved report is missing the run comparison panel"
    )

    logger.info("Saved prediction report: %s", report.url)
    return report.url
