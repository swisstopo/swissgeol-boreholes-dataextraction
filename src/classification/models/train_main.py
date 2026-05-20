"""Unified CLI entry point for BERT classification training.

Usage examples:
    # Train color classifier (conso → unconso)
    fine-tune-bert train --system color --train-conso conso --test-conso unconso

    # Run all 6 consolidation experiments
    fine-tune-bert consolidation-study -cf bert_config_color.yml

    # Train lithology classifier
    fine-tune-bert train --system lithology -cf bert_config_lithology.yml
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import click

from classification import DATAPATH
from classification.models.data_provider import (
    ColorDataProvider,
    LithologyDataProvider,
    TrainingDataProvider,
    consolidation_study_providers,
)
from classification.models.train import BertTrainer, TrainingResult
from classification.utils.classification_classes import ExistingClassificationSystems
from classification.utils.file_utils import read_params
from core.benchmark_utils import configure_logging

_COLOR_OUT_DIR = DATAPATH / "output_color_classification"
_LITHO_OUT_DIR = DATAPATH / "output_classification"

_SUMMARY_METRICS = ["accuracy", "micro_f1", "micro_precision", "micro_recall"]

_BERT_DEFAULTS: dict = {
    "model_path": "google-bert/bert-base-multilingual-uncased",
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.001,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 5.0,
    "max_length": 128,
    "unfreeze_layers": ["classifier", "pooler"],
    "use_class_balancing": False,
    "use_balanced_sampler": True,
}


def _merge(cli_value, cfg: dict, key: str):
    if cli_value is not None:
        return cli_value
    return cfg.get(key, _BERT_DEFAULTS.get(key))


def _load_cfg(config_file: str | None) -> dict:
    if config_file is None:
        return {}
    name = config_file if config_file.endswith(".yml") else f"{config_file}.yml"
    return read_params(f"bert/{name}")


def _build_trainer(
    provider: TrainingDataProvider,
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
    unfreeze_layers: tuple[str, ...] | None,
    use_class_balancing: bool | None,
    use_balanced_sampler: bool | None,
) -> BertTrainer:
    resolved_unfreeze = list(unfreeze_layers) if unfreeze_layers else _merge(None, cfg, "unfreeze_layers")
    return BertTrainer(
        provider=provider,
        model_path=_merge(str(model_path) if model_path else None, cfg, "model_path"),
        out_directory=Path(_merge(str(out_directory) if out_directory else None, cfg, "out_directory")),
        unfreeze_layers=resolved_unfreeze,
        batch_size=_merge(batch_size, cfg, "batch_size"),
        num_epochs=_merge(num_epochs, cfg, "num_epochs"),
        learning_rate=_merge(learning_rate, cfg, "learning_rate"),
        weight_decay=_merge(weight_decay, cfg, "weight_decay"),
        warmup_ratio=_merge(warmup_ratio, cfg, "warmup_ratio"),
        lr_scheduler_type=_merge(lr_scheduler_type, cfg, "lr_scheduler_type"),
        max_grad_norm=_merge(max_grad_norm, cfg, "max_grad_norm"),
        max_length=_merge(max_length, cfg, "max_length"),
        use_class_balancing=_merge(use_class_balancing, cfg, "use_class_balancing"),
        use_balanced_sampler=_merge(use_balanced_sampler, cfg, "use_balanced_sampler"),
    )


def _training_options(f):
    f = click.option(
        "-cf",
        "--config-file",
        type=str,
        default=None,
        help="YAML config file inside config/bert/ (values act as defaults).",
    )(f)
    f = click.option("-m", "--model-path", type=click.Path(path_type=Path), default=None)(f)
    f = click.option("-o", "--out-directory", type=click.Path(path_type=Path), default=None)(f)
    f = click.option("--batch-size", default=None, type=int)(f)
    f = click.option("--num-epochs", default=None, type=int)(f)
    f = click.option("--learning-rate", default=None, type=float)(f)
    f = click.option("--weight-decay", default=None, type=float)(f)
    f = click.option("--warmup-ratio", default=None, type=float)(f)
    f = click.option("--lr-scheduler-type", default=None, type=str)(f)
    f = click.option("--max-grad-norm", default=None, type=float)(f)
    f = click.option("--max-length", default=None, type=int)(f)
    f = click.option(
        "--unfreeze-layers",
        default=None,
        multiple=True,
        help="Layers to unfreeze (repeatable). E.g. --unfreeze-layers classifier --unfreeze-layers pooler",
    )(f)
    f = click.option("--use-class-balancing/--no-use-class-balancing", default=None)(f)
    f = click.option("--use-balanced-sampler/--no-use-balanced-sampler", default=None)(f)
    return f


def _print_summary(results: list[TrainingResult]) -> None:
    col_w = 26
    header = f"{'Experiment':<{col_w}}" + "".join(f"{k:>18}" for k in _SUMMARY_METRICS)
    click.echo("\n" + "=" * len(header))
    click.echo("Training results")
    click.echo("=" * len(header))
    click.echo(header)
    click.echo("-" * len(header))
    for r in results:
        row = f"{r.name:<{col_w}}"
        for k in _SUMMARY_METRICS:
            val = r.metrics.get(k)
            row += f"{val:>18.4f}" if val is not None else f"{'—':>18}"
        click.echo(row)
    click.echo("=" * len(header) + "\n")


def _save_summary_csv(results: list[TrainingResult], out_directory: Path) -> None:
    csv_path = out_directory / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment"] + _SUMMARY_METRICS)
        writer.writeheader()
        for r in results:
            writer.writerow({"experiment": r.name, **{k: r.metrics.get(k, "") for k in _SUMMARY_METRICS}})
    click.echo(f"Results saved to {csv_path}")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """BERT classification training commands."""


# ---------------------------------------------------------------------------
# fine-tune-bert train
# ---------------------------------------------------------------------------


@cli.command("train")
@click.option(
    "-s",
    "--system",
    type=click.Choice(["color", "lithology", "uscs", "en_main"], case_sensitive=False),
    required=True,
    help="Classification system to train.",
)
@click.option(
    "--train-conso",
    type=click.Choice(["conso", "unconso", "mixed"]),
    default="conso",
    help="[color only] Consolidation type to train on.",
)
@click.option(
    "--test-conso",
    type=click.Choice(["conso", "unconso"]),
    default="conso",
    help="[color only] Consolidation type to test on.",
)
@click.option("-n", "--name", default=None, type=str, help="Experiment name for the output directory.")
@_training_options
def train_command(
    system: str,
    train_conso: str,
    test_conso: str,
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
    unfreeze_layers: tuple[str, ...],
    use_class_balancing: bool | None,
    use_balanced_sampler: bool | None,
) -> None:
    """Train a BERT model for a given classification system."""
    configure_logging()
    cfg = _load_cfg(config_file)

    _CONSO_MAP = {"conso": 1, "unconso": 0, "mixed": None}

    if system == "color":
        exp_name = name or f"{train_conso}_to_{test_conso}"
        provider: TrainingDataProvider = ColorDataProvider(
            train_consolidation=_CONSO_MAP[train_conso],
            test_consolidation=_CONSO_MAP[test_conso],
            name=exp_name,
        )
        default_out = _COLOR_OUT_DIR
    else:
        classification_system = ExistingClassificationSystems.get_classification_system_type(system)
        provider = LithologyDataProvider(
            config=cfg,
            classification_system=classification_system,
            name=name or system,
        )
        default_out = _LITHO_OUT_DIR

    cfg.setdefault("out_directory", str(default_out))

    result = _build_trainer(
        provider,
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
        unfreeze_layers or None,
        use_class_balancing,
        use_balanced_sampler,
    ).run()
    _print_summary([result])


# ---------------------------------------------------------------------------
# fine-tune-bert consolidation-study  (color only)
# ---------------------------------------------------------------------------


@cli.command("consolidation-study")
@_training_options
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
    unfreeze_layers: tuple[str, ...],
    use_class_balancing: bool | None,
    use_balanced_sampler: bool | None,
) -> None:
    r"""Run 3 consolidation experiments and print a comparison table.

    \b
    Experiments:
        conso   → conso    (in-distribution baseline)
        unconso → unconso  (in-distribution baseline)
        mixed   → both     (train once on conso+unconso; test on both; unconso metrics prefixed with "unconso_")
    """
    configure_logging()
    cfg = _load_cfg(config_file)
    cfg.setdefault("out_directory", str(_COLOR_OUT_DIR))

    resolved_out = Path(_merge(str(out_directory) if out_directory else None, cfg, "out_directory"))
    results: list[TrainingResult] = []

    for i, provider in enumerate(consolidation_study_providers(), start=1):
        logging.getLogger(__name__).info("[%d/3] Running experiment: %s", i, provider.name)
        result = _build_trainer(
            provider,
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
            unfreeze_layers or None,
            use_class_balancing,
            use_balanced_sampler,
        ).run()
        results.append(result)

    _print_summary(results)
    _save_summary_csv(results, resolved_out)


if __name__ == "__main__":
    cli()
