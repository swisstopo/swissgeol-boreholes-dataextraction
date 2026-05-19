"""CLI entry point for color classification training."""

import csv
import logging
from pathlib import Path

import click

from classification import DATAPATH
from classification.color.dataset_config import DATASET_REGISTRY, ColorDatasetConfig, consolidation_study_configs
from classification.color.runner import ColorTrainingResult, ColorTrainingRunner
from classification.utils.file_utils import read_params
from core.benchmark_utils import configure_logging

# Hardcoded fallback defaults (used when neither YAML nor CLI provides a value).
_SUMMARY_METRICS = ["accuracy", "micro_f1", "micro_precision", "micro_recall"]

_DEFAULTS = {
    "model_path": "google-bert/bert-base-multilingual-uncased",
    "out_directory": str(DATAPATH / "output_color_classification"),
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.001,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 5.0,
    "max_length": 128,
    "use_class_balancing": False,
}


def _merge(cli_value, cfg: dict, key: str):
    """Return cli_value if explicitly set (not None), else cfg value, else _DEFAULTS value."""
    if cli_value is not None:
        return cli_value
    return cfg.get(key, _DEFAULTS[key])


def _load_cfg(config_file: str | None) -> dict:
    if config_file is None:
        return {}
    name = config_file if config_file.endswith(".yml") else f"{config_file}.yml"
    return read_params(f"bert/{name}")


def _runner_from_options(
    config: ColorDatasetConfig,
    cfg: dict,
    model_path: Path | None,
    out_directory: Path | None,
    batch_size: int | None,
    num_epochs: int | None,
    learning_rate: float | None,
    weight_decay: float | None,
    warmup_ratio: float | None,
    lr_scheduler_type: str | None,
    max_grad_norm: float | None,
    max_length: int | None,
    use_class_balancing: bool | None,
) -> ColorTrainingRunner:
    return ColorTrainingRunner(
        config=config,
        model_path=_merge(model_path, cfg, "model_path"),
        out_directory=Path(_merge(str(out_directory) if out_directory else None, cfg, "out_directory")),
        batch_size=_merge(batch_size, cfg, "batch_size"),
        num_epochs=_merge(num_epochs, cfg, "num_epochs"),
        learning_rate=_merge(learning_rate, cfg, "learning_rate"),
        weight_decay=_merge(weight_decay, cfg, "weight_decay"),
        warmup_ratio=_merge(warmup_ratio, cfg, "warmup_ratio"),
        lr_scheduler_type=_merge(lr_scheduler_type, cfg, "lr_scheduler_type"),
        max_grad_norm=_merge(max_grad_norm, cfg, "max_grad_norm"),
        max_length=_merge(max_length, cfg, "max_length"),
        use_class_balancing=_merge(use_class_balancing, cfg, "use_class_balancing"),
    )


def _print_summary(results: list[ColorTrainingResult]) -> None:
    col_w = 26
    header = f"{'Experiment':<{col_w}}" + "".join(f"{k:>18}" for k in _SUMMARY_METRICS)
    click.echo("\n" + "=" * len(header))
    click.echo("Consolidation study — results")
    click.echo("=" * len(header))
    click.echo(header)
    click.echo("-" * len(header))
    for result in results:
        row = f"{result.config.name:<{col_w}}"
        for k in _SUMMARY_METRICS:
            val = result.metrics.get(k)
            row += f"{val:>18.4f}" if val is not None else f"{'—':>18}"
        click.echo(row)
    click.echo("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Shared CLI options
# ---------------------------------------------------------------------------


def _training_options(f):
    f = click.option(
        "-cf",
        "--config-file",
        type=str,
        default=None,
        help="YAML config file name inside config/bert/ (e.g. bert_config_color.yml). "
        "Values from the file act as defaults; explicit CLI flags take precedence.",
    )(f)
    f = click.option(
        "-m",
        "--model-path",
        type=click.Path(path_type=Path),
        default=None,
        help="Local BERT checkpoint path or HuggingFace model ID.",
    )(f)
    f = click.option(
        "-o",
        "--out-directory",
        type=click.Path(path_type=Path),
        default=None,
        help="Root directory for model checkpoints and logs.",
    )(f)
    f = click.option("--batch-size", default=None, type=int, help="Per-device batch size.")(f)
    f = click.option("--num-epochs", default=None, type=int, help="Number of training epochs.")(f)
    f = click.option("--learning-rate", default=None, type=float, help="Peak learning rate.")(f)
    f = click.option("--weight-decay", default=None, type=float, help="L2 regularization weight.")(f)
    f = click.option("--warmup-ratio", default=None, type=float, help="Fraction of steps for LR warm-up.")(f)
    f = click.option("--lr-scheduler-type", default=None, type=str, help="LR scheduler type (e.g. cosine, linear).")(f)
    f = click.option("--max-grad-norm", default=None, type=float, help="Gradient clipping threshold.")(f)
    f = click.option(
        "--max-length", default=None, type=int, help="Maximum token sequence length (reduce to save GPU memory)."
    )(f)
    f = click.option(
        "--use-class-balancing/--no-use-class-balancing",
        default=None,
        help="Weight loss by inverse class frequency (default: from config).",
    )(f)
    return f


def _list_slices_option(f):
    return click.option(
        "--list-slices",
        is_flag=True,
        default=False,
        is_eager=True,
        expose_value=False,
        callback=lambda ctx, _param, value: (
            click.echo("Available dataset slices:\n  " + "\n  ".join(DATASET_REGISTRY)) or ctx.exit()
        )
        if value
        else None,
        help="Print all available dataset slice names and exit.",
    )(f)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Color classification training commands."""


@cli.command("train")
@click.option(
    "-t",
    "--train",
    "train_slices",
    multiple=True,
    help="Slice(s) to train on (repeatable). E.g. -t TH-conso -t NA-conso.",
)
@click.option(
    "-e", "--test", "test_slices", multiple=True, help="Slice(s) to evaluate on (repeatable). E.g. -e TH-unconso."
)
@click.option("-n", "--name", default=None, type=str, help="Experiment name for output directory naming and MLflow.")
@_training_options
@_list_slices_option
def train_command(
    train_slices: tuple[str, ...],
    test_slices: tuple[str, ...],
    name: str | None,
    config_file: str | None,
    model_path: Path | None,
    out_directory: Path | None,
    batch_size: int | None,
    num_epochs: int | None,
    learning_rate: float | None,
    weight_decay: float | None,
    warmup_ratio: float | None,
    lr_scheduler_type: str | None,
    max_grad_norm: float | None,
    max_length: int | None,
    use_class_balancing: bool | None,
) -> None:
    r"""Train a BERT model to predict color from borehole layer descriptions.

    Hyperparameters can come from a YAML config file (-cf) with individual
    flags overriding the file values. Train/test slices can also be specified
    in the YAML config under train_slices / test_slices.

    Examples:
    \b
        # Train using all settings from the default config file
        boreholes-train-color train -cf bert_config_color.yml

    \b
        # Override epochs from the config file
        boreholes-train-color train -cf bert_config_color.yml --num-epochs 5

    \b
        # Train on Thurgau consolidated, test on Thurgau unconsolidated (no config file)
        boreholes-train-color train -t TH-conso -e TH-unconso

    \b
        # Pool multiple sources for training
        boreholes-train-color train -t ZH-conso -t TH-conso -e NA-unconso -n zh_th_to_na
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    cfg = _load_cfg(config_file)

    resolved_train = list(train_slices) or cfg.get("train_slices", [])
    resolved_test = list(test_slices) or cfg.get("test_slices", [])
    resolved_name = name or cfg.get("name", "color_experiment")

    if not resolved_train or not resolved_test:
        raise click.UsageError(
            "Train and test slices are required. Provide -t / -e flags or set train_slices / "
            "test_slices in the YAML config file."
        )

    config = ColorDatasetConfig(resolved_train, resolved_test, resolved_name)
    logger.info("Starting color training — train: %s | test: %s", config.train_slices, config.test_slices)

    result = _runner_from_options(
        config,
        cfg,
        model_path,
        out_directory,
        batch_size,
        num_epochs,
        learning_rate,
        weight_decay,
        warmup_ratio,
        lr_scheduler_type,
        max_grad_norm,
        max_length,
        use_class_balancing,
    ).run()
    _print_summary([result])


@cli.command("consolidation-study")
@_training_options
@_list_slices_option
def consolidation_study_command(
    config_file: str | None,
    model_path: Path | None,
    out_directory: Path | None,
    batch_size: int | None,
    num_epochs: int | None,
    learning_rate: float | None,
    weight_decay: float | None,
    warmup_ratio: float | None,
    lr_scheduler_type: str | None,
    max_grad_norm: float | None,
    max_length: int | None,
    use_class_balancing: bool | None,
) -> None:
    r"""Run all 6 consolidation generalisation experiments and print a comparison table.

    Experiments (all using the full cross-dataset pool):
    \b
        conso   -> conso    (in-distribution baseline)
        conso   -> unconso  (cross-type generalisation)
        unconso -> conso    (cross-type generalisation)
        unconso -> unconso  (in-distribution baseline)
        mixed   -> conso    (does mixed training help on conso?)
        mixed   -> unconso  (does mixed training help on unconso?)
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    cfg = _load_cfg(config_file)

    configs = consolidation_study_configs()
    results: list[ColorTrainingResult] = []

    for i, config in enumerate(configs, start=1):
        logger.info("[%d/%d] Running experiment: %s", i, len(configs), config.name)
        result = _runner_from_options(
            config,
            cfg,
            model_path,
            out_directory,
            batch_size,
            num_epochs,
            learning_rate,
            weight_decay,
            warmup_ratio,
            lr_scheduler_type,
            max_grad_norm,
            max_length,
            use_class_balancing,
        ).run()
        results.append(result)

    resolved_out = Path(_merge(str(out_directory) if out_directory else None, cfg, "out_directory"))
    _print_summary(results)
    _save_summary_csv(results, resolved_out)


def _save_summary_csv(results: list[ColorTrainingResult], out_directory: Path) -> None:
    csv_path = out_directory / "consolidation_study_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment"] + _SUMMARY_METRICS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {"experiment": result.config.name, **{k: result.metrics.get(k, "") for k in _SUMMARY_METRICS}}
            )
    click.echo(f"Results saved to {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Keep the top-level name `click_pipeline` so the pyproject.toml entry point
# (`boreholes-train-color = "classification.color.main:click_pipeline"`) still works.
click_pipeline = cli

if __name__ == "__main__":
    cli()
