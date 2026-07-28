"""Build a W&B report comparing a prediction run against a baseline run.

Uses `wandb.Api()` to validate the two runs, read metadata for the report text, and pull
each run's logged `png_browser_table` to build a side-by-side image comparison table.
The side-by-side table is logged onto whichever run is already active, which the caller owns.
"""

import logging
import tempfile
from pathlib import Path

import wandb_workspaces.expr as expr
import wandb_workspaces.reports.v2 as wr

import wandb
from reports.comparison_table import build_comparison_table, load_table

logger = logging.getLogger(__name__)

COMPARISON_TABLE_KEY = "prediction_comparison_table"
PNG_BROWSER_TABLE_KEY = "png_browser_table"


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

    with tempfile.TemporaryDirectory() as tmp_dir:
        table, gallery, *_ = build_comparison_table(
            rows_base,
            dir_base,
            rows_pred,
            dir_pred,
            join_key="filename",
            image_column="image",
            run_a_id=baseline_run_id,
            run_b_id=prediction_run_id,
            out_dir=Path(tmp_dir),
        )
        wandb.log({COMPARISON_TABLE_KEY: table, "prediction_comparison_gallery": gallery})
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
    if comparison_table_logged:
        summary_lines.append(f'- Tip: filter `{COMPARISON_TABLE_KEY}` below by filename using `col0 == "<filename>"`.')
    summary_md = "\n".join(summary_lines)

    blocks = [
        wr.H1(text=title),
        wr.MarkdownBlock(text=summary_md),
        wr.PanelGrid(
            runsets=[baseline_runset, prediction_runset],
            panels=[wr.RunComparer(layout=wr.Layout(w=24, h=12))],
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
            blocks.append(
                wr.PanelGrid(
                    runsets=[baseline_runset, prediction_runset],
                    panels=[wr.BarPlot(metrics=present_metrics, layout=wr.Layout(w=24, h=9))],
                )
            )

    if comparison_table_logged:
        blocks.append(
            wr.PanelGrid(
                runsets=[prediction_runset],
                panels=[wr.WeavePanelSummaryTable(table_name=COMPARISON_TABLE_KEY, layout=wr.Layout(w=24, h=16))],
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
