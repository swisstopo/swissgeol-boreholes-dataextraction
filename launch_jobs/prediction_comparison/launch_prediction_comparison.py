"""W&B Launch entrypoint for the prediction comparison job.

Config-driven twin of wandb_prediction_comparison.py: instead of Click options, all
inputs come from wandb.config so this can run as a W&B Launch job with run_a/run_b
overridden per launch, without touching the code. Reuses the same core functions
(stack_side_by_side, find_table_artifact, load_table, build_comparison_table).

Flat-import copy of src/scripts/launch_prediction_comparison.py, kept in sync by hand,
so this directory is self-contained (no dependency on the rest of the repo) and cheap
to upload as a W&B Launch "code" job.

Run locally with, e.g.:
    WANDB_ENTITY=... WANDB_PROJECT=... python launch_prediction_comparison.py
(fails fast if run_a/run_b aren't supplied via a Launch config override).
"""

import os
from pathlib import Path

from wandb_prediction_comparison import build_comparison_table, load_table
from wandb_tracking import wandb, wandb_tracking

DEFAULT_CONFIG = {
    "entity": os.getenv("WANDB_ENTITY", "swisstopo-visium"),
    "project": os.getenv("WANDB_PROJECT", "swissgeol-boreholes"),
    "run_a": None,
    "run_b": None,
    "table_key": "png_browser_table",
    "join_key": "filename",
    "image_column": "image",
    "comparison_run_name": "prediction-comparison",
    "out_dir": "data/output/wandb_comparison",
}


def main():
    """Fetch a table from two W&B runs (per wandb.config) and log a side-by-side image comparison."""
    if not wandb_tracking or wandb is None:
        raise RuntimeError("Set WANDB_TRACKING=True (and run `uv sync --extra wandb`) before using this script.")

    wandb.init(
        entity=DEFAULT_CONFIG["entity"],
        project=DEFAULT_CONFIG["project"],
        name=DEFAULT_CONFIG["comparison_run_name"],
        job_type="comparison",
        config=DEFAULT_CONFIG,
    )
    try:
        cfg = wandb.config
        if not cfg.run_a or not cfg.run_b:
            raise ValueError("Both config values 'run_a' and 'run_b' are required.")
        wandb.run.name = cfg.comparison_run_name

        api = wandb.Api(timeout=120)
        run_a_obj = api.run(f"{cfg.entity}/{cfg.project}/{cfg.run_a}")
        run_b_obj = api.run(f"{cfg.entity}/{cfg.project}/{cfg.run_b}")

        rows_a, dir_a = load_table(run_a_obj, cfg.table_key)
        rows_b, dir_b = load_table(run_b_obj, cfg.table_key)

        table, gallery, shared_keys, only_in_a, only_in_b = build_comparison_table(
            rows_a,
            dir_a,
            rows_b,
            dir_b,
            join_key=cfg.join_key,
            image_column=cfg.image_column,
            run_a_id=cfg.run_a,
            run_b_id=cfg.run_b,
            out_dir=Path(cfg.out_dir),
        )

        wandb.log(
            {
                "prediction_comparison_table": table,
                "large_image_comparison": gallery,
                "num_shared_files": len(shared_keys),
                "num_only_in_a": len(only_in_a),
                "num_only_in_b": len(only_in_b),
            }
        )
        wandb.summary["run_a"] = cfg.run_a
        wandb.summary["run_b"] = cfg.run_b
        wandb.summary["table_key"] = cfg.table_key
        wandb.summary["join_key"] = cfg.join_key
        wandb.summary["num_shared_files"] = len(shared_keys)
        wandb.summary["num_only_in_a"] = len(only_in_a)
        wandb.summary["num_only_in_b"] = len(only_in_b)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
