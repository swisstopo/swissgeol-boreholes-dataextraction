"""Build a side-by-side image comparison across two W&B runs' png_browser_table.

For every row that both runs logged under the same join key, stitches the two
prediction images into one wide PNG (labeled by run id) and logs a new comparison
run with a searchable Table (one row per file) and a media gallery for browsing.

Flat-import copy of src/scripts/wandb_prediction_comparison.py, kept in sync by hand,
for use as a self-contained W&B Launch "code" job (see launch_prediction_comparison.py).
"""

import json
import logging
import os
from pathlib import Path

import click
import cv2
import numpy as np
from wandb_tracking import wandb, wandb_tracking

logger = logging.getLogger(__name__)


def stack_side_by_side(image_paths: list[Path], labels: list[str]) -> np.ndarray:
    """Resize images to a common height and stack them horizontally, each labeled with its run.

    Args:
        image_paths (list[Path]): Images to combine, left to right.
        labels (list[str]): One label per image (e.g. the run id), drawn in its top-left corner.

    Returns:
        np.ndarray: The combined BGR image.
    """
    images = [cv2.imread(str(p)) for p in image_paths]
    missing = [p for p, img in zip(image_paths, images, strict=True) if img is None]
    if missing:
        raise ValueError(f"Could not read image(s): {missing}")

    target_height = min(img.shape[0] for img in images)
    labeled = []
    for img, label in zip(images, labels, strict=True):
        scale = target_height / img.shape[0]
        resized = cv2.resize(img, (int(img.shape[1] * scale), target_height))
        cv2.putText(resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        labeled.append(resized)

    divider = np.full((target_height, 4, 3), 255, dtype=np.uint8)
    stacked = labeled[0]
    for img in labeled[1:]:
        stacked = np.hstack([stacked, divider, img])
    return stacked


def find_table_artifact(run, table_key: str):
    """Find the artifact a run logged for the given wandb.log() table key."""
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
    out_dir: Path,
) -> tuple["wandb.Table", list, list[str], set[str], set[str]]:
    """Join two tables' rows by `join_key` and build a side-by-side comparison Table + gallery.

    Args:
        rows_a (list[dict]): Rows from the first run's table (see `load_table`).
        dir_a (Path): Local artifact dir for `rows_a`, used to resolve its image paths.
        rows_b (list[dict]): Rows from the second run's table.
        dir_b (Path): Local artifact dir for `rows_b`.
        join_key (str): Column shared by both tables to match rows on (e.g. "filename").
        image_column (str): Column holding the W&B Image cell to combine (e.g. "image").
        run_a_id (str): First run's id, used as a label and table column value.
        run_b_id (str): Second run's id, used as a label and table column value.
        out_dir (Path): Where to write the generated side-by-side PNGs.

    Returns:
        tuple: (table, gallery, shared_keys, only_in_a, only_in_b).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key_a = {row[join_key]: row for row in rows_a}
    by_key_b = {row[join_key]: row for row in rows_b}

    shared_keys = sorted(by_key_a.keys() & by_key_b.keys())
    only_in_a = by_key_a.keys() - by_key_b.keys()
    only_in_b = by_key_b.keys() - by_key_a.keys()
    if only_in_a or only_in_b:
        logger.warning(
            "Skipping %d file(s) only in %s and %d only in %s (not present in both runs)",
            len(only_in_a),
            run_a_id,
            len(only_in_b),
            run_b_id,
        )

    table = wandb.Table(columns=[join_key, "comparison_image", "run_a_id", "run_b_id"])
    gallery = []
    for key in shared_keys:
        image_a = _image_path(by_key_a[key], image_column, dir_a)
        image_b = _image_path(by_key_b[key], image_column, dir_b)
        combined = stack_side_by_side([image_a, image_b], labels=[run_a_id, run_b_id])

        safe_name = key.replace("/", "_")
        combined_path = out_dir / f"{safe_name}_comparison.png"
        cv2.imwrite(str(combined_path), combined)

        table.add_data(key, wandb.Image(str(combined_path), caption=key), run_a_id, run_b_id)
        gallery.append(wandb.Image(str(combined_path), caption=key))

    return table, gallery, shared_keys, only_in_a, only_in_b


@click.command()
@click.option("--project", default=None, help="W&B '[entity/]project' (defaults to $WANDB_PROJECT).")
@click.option("--run-a", required=True, help="First run id.")
@click.option("--run-b", required=True, help="Second run id.")
@click.option("--table-key", default="png_browser_table", help="Table log key to compare.")
@click.option("--join-key", default="filename", help="Column shared by both tables to match rows on.")
@click.option("--image-column", default="image", help="Column holding the W&B Image cell to combine.")
@click.option(
    "--out-dir",
    default="data/output/wandb_comparison",
    type=click.Path(path_type=Path),
    help="Where to write the generated side-by-side PNGs.",
)
@click.option("--comparison-run-name", default="prediction-comparison")
def main(
    project: str | None,
    run_a: str,
    run_b: str,
    table_key: str,
    join_key: str,
    image_column: str,
    out_dir: Path,
    comparison_run_name: str,
):
    """Fetch a table from two W&B runs and log a side-by-side image comparison.

    \f
    Args:
        project (str | None): W&B "[entity/]project", defaults to $WANDB_PROJECT.
        run_a (str): First run id.
        run_b (str): Second run id.
        table_key (str): Table log key to compare (e.g. "png_browser_table").
        join_key (str): Column shared by both tables to match rows on (e.g. "filename").
        image_column (str): Column holding the W&B Image cell to combine (e.g. "image").
        out_dir (Path): Where to write the generated side-by-side PNGs.
        comparison_run_name (str): Name for the new W&B run that logs the comparison.
    """  # noqa: D301
    if not wandb_tracking or wandb is None:
        raise RuntimeError("Set WANDB_TRACKING=True (and run `uv sync --extra wandb`) before using this script.")

    project = project or os.getenv("WANDB_PROJECT", "swissgeol-boreholes")

    api = wandb.Api()
    run_a_obj = api.run(f"{project}/{run_a}")
    run_b_obj = api.run(f"{project}/{run_b}")

    rows_a, dir_a = load_table(run_a_obj, table_key)
    rows_b, dir_b = load_table(run_b_obj, table_key)

    wandb.init(
        project=project, name=comparison_run_name, job_type="comparison", config={"run_a": run_a, "run_b": run_b}
    )
    try:
        table, gallery, _, _, _ = build_comparison_table(
            rows_a,
            dir_a,
            rows_b,
            dir_b,
            join_key=join_key,
            image_column=image_column,
            run_a_id=run_a,
            run_b_id=run_b,
            out_dir=out_dir,
        )
        wandb.log({"prediction_comparison_table": table, "large_image_comparison": gallery})
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
