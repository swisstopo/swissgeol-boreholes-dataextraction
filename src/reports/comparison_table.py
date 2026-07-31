"""Build a side-by-side image comparison across two W&B runs' png_browser_table.

For every row that both runs logged under the same join key, builds one Table row holding
each run's own image in its own column (named after that run's id) plus a media gallery,
so the W&B UI renders both runs' images side by side without any local image processing.
"""

import json
import logging
from pathlib import Path

import backoff

import wandb

logger = logging.getLogger(__name__)


@backoff.on_exception(backoff.expo, ValueError, max_time=20)
def find_table_artifact(run, table_key: str):
    """Find the artifact a run logged for the given wandb.log() table key.

    Retries for a while: a run's own table artifact can still be uploading in the
    background when queried right after `wandb.log()`, before it's visible via the API.
    """
    matches = [a for a in run.logged_artifacts() if table_key in a.name]
    if not matches:
        raise ValueError(f"Run {run.id} has no logged artifact matching table key {table_key!r}")
    if len(matches) > 1:
        logger.warning("Run %s logged %d artifacts matching %r, using the last one", run.id, len(matches), table_key)
    return matches[-1]


def load_table(run, table_key: str) -> tuple[list[dict], Path]:
    """Download a run's table artifact and return its rows (as dicts) plus the local artifact dir."""
    artifact = find_table_artifact(run, table_key)
    table_dir = Path(artifact.download())
    table_json_path = next(table_dir.rglob("*.table.json"))
    with open(table_json_path, encoding="utf8") as f:
        raw = json.load(f)
    rows = [dict(zip(raw["columns"], row, strict=True)) for row in raw["data"]]
    return rows, table_dir


def _image_path(row: dict, image_column: str, table_dir: Path) -> Path:
    cell = row.get(image_column)
    if not isinstance(cell, dict) or "path" not in cell:
        raise ValueError("Expected row[image_column] to be a W&B Image dict with a 'path' field.")
    return table_dir / cell["path"]


def build_multi_run_comparison_table(
    runs: dict[str, tuple[list[dict], Path]],
    *,
    join_key: str,
    image_column: str,
    max_images: int | None = None,
) -> tuple["wandb.Table", list, list[str], dict[str, set[str]]]:
    """Join two or more runs' rows by `join_key` into a table with one image column per run.

    Args:
        runs (dict[str, tuple[list[dict], Path]]): Mapping of run id to that run's
            (rows, table_dir), as returned by `load_table`. One entry per run being compared.
        join_key (str): Column shared by every run's table to match rows on (e.g. "filename").
        image_column (str): Column holding the W&B Image cell to read (e.g. "image").
        max_images (int | None): Cap on the number of shared rows to include. `None` for no cap.

    Returns:
        tuple: (table, gallery, shared_keys, only_in_run). `only_in_run` maps each run id to the
        keys present in that run but missing from at least one other (excluded from the table).
    """
    keys_by_run = {run_id: {row[join_key] for row in rows} for run_id, (rows, _) in runs.items()}
    shared_keys = sorted(set.intersection(*keys_by_run.values()))
    only_in_run = {run_id: keys - set(shared_keys) for run_id, keys in keys_by_run.items()}
    for run_id, missing in only_in_run.items():
        if missing:
            logger.warning("Skipping %d file(s) only in %s (not present in every compared run)", len(missing), run_id)

    if max_images is not None and len(shared_keys) > max_images:
        logger.warning("Capping comparison table at %d of %d shared file(s)", max_images, len(shared_keys))
        shared_keys = shared_keys[:max_images]

    by_key_per_run = {run_id: {row[join_key]: row for row in rows} for run_id, (rows, _) in runs.items()}

    table = wandb.Table(columns=[join_key, *runs])
    gallery = []
    for key in shared_keys:
        images = []
        for run_id, (_, table_dir) in runs.items():
            row = by_key_per_run[run_id][key]
            image = wandb.Image(str(_image_path(row, image_column, table_dir)), caption=f"{key} ({run_id})")
            images.append(image)
            gallery.append(image)
        table.add_data(key, *images)

    return table, gallery, shared_keys, only_in_run


def build_comparison_table(
    rows_a: list[dict],
    dir_a: Path,
    rows_b: list[dict],
    dir_b: Path,
    *,
    join_key: str,
    image_column: str,
    run_a_id: str,
    run_b_id: str,
) -> tuple["wandb.Table", list, list[str], set[str], set[str]]:
    """Join two tables' rows by `join_key` into a table with one image column per run.

    Args:
        rows_a (list[dict]): Rows from the first run's table (see `load_table`).
        dir_a (Path): Local artifact dir for `rows_a`, used to resolve its image paths.
        rows_b (list[dict]): Rows from the second run's table.
        dir_b (Path): Local artifact dir for `rows_b`.
        join_key (str): Column shared by both tables to match rows on (e.g. "filename").
        image_column (str): Column holding the W&B Image cell to read (e.g. "image").
        run_a_id (str): First run's id, used as both the table column name and image caption.
        run_b_id (str): Second run's id, used as both the table column name and image caption.

    Returns:
        tuple: (table, gallery, shared_keys, only_in_a, only_in_b).
    """
    table, gallery, shared_keys, only_in_run = build_multi_run_comparison_table(
        {run_a_id: (rows_a, dir_a), run_b_id: (rows_b, dir_b)},
        join_key=join_key,
        image_column=image_column,
    )
    return table, gallery, shared_keys, only_in_run[run_a_id], only_in_run[run_b_id]


def build_run_gallery_table(
    runs: dict[str, tuple[list[dict], Path]],
    *,
    join_key: str,
    image_column: str,
    max_images: int | None = None,
) -> tuple["wandb.Table", list, list[str]]:
    """Union any number of runs' rows by `join_key` into one table, one image column per run.

    This keeps every file logged by any included run - a run's
    cell is just left blank for files it didn't process.

    Args:
        runs (dict[str, tuple[list[dict], Path]]): Mapping of run id to that run's
            (rows, table_dir), as returned by `load_table`.
        join_key (str): Column shared by every run's table to match rows on (e.g. "filename").
        image_column (str): Column holding the W&B Image cell to read (e.g. "image").
        max_images (int | None): Cap on the number of rows to include. `None` for no cap.

    Returns:
        tuple: (table, gallery, keys) - every file key included, across every run.
    """
    by_key_per_run = {run_id: {row[join_key]: row for row in rows} for run_id, (rows, _) in runs.items()}
    all_keys = sorted(set.union(*(set(rows_by_key) for rows_by_key in by_key_per_run.values())))

    if max_images is not None and len(all_keys) > max_images:
        logger.warning("Capping gallery table at %d of %d file(s)", max_images, len(all_keys))
        all_keys = all_keys[:max_images]

    table = wandb.Table(columns=[join_key, *runs])
    gallery = []
    for key in all_keys:
        images = []
        for run_id, (_, table_dir) in runs.items():
            row = by_key_per_run[run_id].get(key)
            image = (
                wandb.Image(str(_image_path(row, image_column, table_dir)), caption=f"{key} ({run_id})")
                if row is not None
                else None
            )
            images.append(image)
            if image is not None:
                gallery.append(image)
        table.add_data(key, *images)

    return table, gallery, all_keys


def dedupe_file_metric_rows(
    rows: list[dict],
    metric_columns: list[str],
    preferred_key_columns: tuple[str, ...] = ("pdf_filename", "page"),
) -> tuple[dict[tuple, dict], list[str]]:
    """Collapse repeated per-PNG rows of `png_browser_table` into one row per file/document.

    `png_browser_table` logs one row per PNG browser entry (e.g. both a `*_outputs.png` and a
    `*_tables.png` row for the same PDF), so several rows can share the same file-level metrics.
    This groups rows by the most specific available file-identifying columns and keeps a single
    representative row per group. If duplicate rows disagree on a metric value, that indicates a
    real data problem (not just harmless PNG repetition), so the group is flagged and a warning
    is logged, while still keeping the first row.

    Args:
        rows (list[dict]): Rows as returned by `load_table`.
        metric_columns (list[str]): Metric column names to check for consistency across duplicates.
        preferred_key_columns (tuple[str, ...]): Candidate key columns to look for, in priority order.
            Used in full only if every column in it is present on the rows; otherwise only its first
            column is tried, and "filename" is used as the last resort.

    Returns:
        tuple: (rows_by_key, key_columns) - a mapping from key tuple to representative row (with an
            added "duplicate_metric_conflict" bool), and the key column names actually used.
    """
    if not rows:
        return {}, [preferred_key_columns[0]] if preferred_key_columns else ["filename"]

    sample = rows[0]
    if all(col in sample for col in preferred_key_columns):
        key_columns = list(preferred_key_columns)
    elif preferred_key_columns and preferred_key_columns[0] in sample:
        key_columns = [preferred_key_columns[0]]
    else:
        key_columns = ["filename"]

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(col) for col in key_columns)
        groups.setdefault(key, []).append(row)

    rows_by_key = {}
    for key, group_rows in groups.items():
        first = group_rows[0]
        conflict = any(other.get(col) != first.get(col) for other in group_rows[1:] for col in metric_columns)
        if conflict:
            logger.warning(
                "Duplicate png_browser_table rows for file key %s have conflicting metric values; "
                "keeping the first row.",
                key,
            )
        rows_by_key[key] = {**first, "duplicate_metric_conflict": conflict}

    return rows_by_key, key_columns


def _metric_direction(
    metric: str, delta: float | None, changed: bool, lower_is_better: set[str], count_metrics: set[str]
) -> str:
    """Classify a single metric's change as "same", "changed", "improved", or "worse"."""
    if delta is None or not changed:
        return "same"
    if metric in count_metrics:
        return "changed"
    if metric in lower_is_better:
        return "improved" if delta < 0 else "worse"
    return "improved" if delta > 0 else "worse"


def build_file_metric_comparison_tables(
    rows_base: list[dict],
    rows_pred: list[dict],
    *,
    metric_columns: list[str],
    run_a_id: str,
    run_b_id: str,
    lower_is_better_metrics: set[str] = frozenset({"layer_num_wrong"}),
    count_metrics: set[str] = frozenset({"layer_num_total", "groundwater_depth_num_detected"}),
    tolerance: float = 1e-12,
) -> tuple["wandb.Table", "wandb.Table", list[tuple], set[tuple], set[tuple]]:
    """Join baseline and prediction file-level metrics (deduped by pdf_filename[+page]) into two Tables.

    Args:
        rows_base (list[dict]): Baseline run's `png_browser_table` rows (see `load_table`).
        rows_pred (list[dict]): Prediction run's `png_browser_table` rows.
        metric_columns (list[str]): Metric column names to compare.
        run_a_id (str): Baseline run id, used as a label and column value.
        run_b_id (str): Prediction run id, used as a label and column value.
        lower_is_better_metrics (set[str]): Metrics where a decrease counts as "improved".
        count_metrics (set[str]): Metrics classified as "changed" rather than improved/worse.
        tolerance (float): Minimum absolute delta to consider a metric changed.

    Returns:
        tuple: (wide_table, long_table, shared_keys, only_in_base, only_in_prediction).
    """
    rows_by_key_base, key_columns_base = dedupe_file_metric_rows(rows_base, metric_columns)
    rows_by_key_pred, key_columns_pred = dedupe_file_metric_rows(rows_pred, metric_columns)
    key_columns = key_columns_base if rows_by_key_base else key_columns_pred

    shared_keys = sorted(rows_by_key_base.keys() & rows_by_key_pred.keys())
    only_in_base = rows_by_key_base.keys() - rows_by_key_pred.keys()
    only_in_prediction = rows_by_key_pred.keys() - rows_by_key_base.keys()
    if only_in_base or only_in_prediction:
        logger.warning(
            "Skipping %d file(s) only in baseline %s and %d only in prediction %s for file-level metric comparison",
            len(only_in_base),
            run_a_id,
            len(only_in_prediction),
            run_b_id,
        )

    wide_columns = [
        *key_columns,
        "run_a_id",
        "run_b_id",
        "num_changed_metrics",
        "mean_abs_delta",
        "max_abs_delta",
        "biggest_changed_metric",
        "duplicate_metric_conflict_base",
        "duplicate_metric_conflict_prediction",
    ]
    for metric in metric_columns:
        wide_columns += [
            f"{metric}_baseline",
            f"{metric}_prediction",
            f"{metric}_delta",
            f"{metric}_abs_delta",
            f"{metric}_changed",
        ]
    wide_table = wandb.Table(columns=wide_columns)

    long_table = wandb.Table(
        columns=[
            *key_columns,
            "metric",
            "baseline_value",
            "prediction_value",
            "delta",
            "abs_delta",
            "changed",
            "direction",
            "run_a_id",
            "run_b_id",
        ]
    )

    for key in shared_keys:
        base_row = rows_by_key_base[key]
        pred_row = rows_by_key_pred[key]

        per_metric_cells = []
        changed_metrics = []
        abs_deltas = []
        for metric in metric_columns:
            base_val = base_row.get(metric)
            pred_val = pred_row.get(metric)
            if base_val is None or pred_val is None:
                delta = None
                abs_delta = None
                changed = False
            else:
                delta = pred_val - base_val
                abs_delta = abs(delta)
                changed = abs_delta > tolerance

            if changed:
                changed_metrics.append(metric)
                abs_deltas.append(abs_delta)

            direction = _metric_direction(metric, delta, changed, lower_is_better_metrics, count_metrics)
            per_metric_cells += [base_val, pred_val, delta, abs_delta, changed]

            long_table.add_data(
                *key, metric, base_val, pred_val, delta, abs_delta, changed, direction, run_a_id, run_b_id
            )

        max_abs_delta = max(abs_deltas) if abs_deltas else 0.0
        biggest_changed_metric = changed_metrics[abs_deltas.index(max_abs_delta)] if abs_deltas else None

        wide_table.add_data(
            *key,
            run_a_id,
            run_b_id,
            len(changed_metrics),
            sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0,
            max_abs_delta,
            biggest_changed_metric,
            bool(base_row["duplicate_metric_conflict"]),
            bool(pred_row["duplicate_metric_conflict"]),
            *per_metric_cells,
        )

    return wide_table, long_table, shared_keys, only_in_base, only_in_prediction
