"""Combine a shared backbone and a task-specific head into a single full BERT model.

Merges the weights from backbone.safetensors and a head directory (model.safetensors +
config/tokenizer files) and writes a self-contained HuggingFace model directory that can
be loaded with AutoModelForSequenceClassification.from_pretrained().

This is the inverse of model_decoupling.py.

Usage:
    python src/scripts/model_coupling.py \
        --backbone-path models/backbone/backbone.safetensors \
        --head-path models/model_heads/en_secondary \
        --output-path models/recoupled_models/en_secondary_full
"""

import shutil
from pathlib import Path

import click
import torch
from safetensors import safe_open
from safetensors.torch import save_file

INFERENCE_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from a safetensors file into a dict."""
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            tensors[key] = f.get_tensor(key)
    return tensors


def couple(backbone_path: Path, head_path: Path, output_path: Path) -> None:
    """Merge backbone and head weights and write a full HuggingFace model directory.

    Args:
        backbone_path: Path to backbone.safetensors.
        head_path: Path to the head directory (must contain model.safetensors and config.json).
        output_path: Directory to write the merged model into.
    """
    backbone_file = backbone_path
    head_file = head_path / "model.safetensors"

    if not backbone_file.exists():
        raise FileNotFoundError(f"Backbone not found: {backbone_file}")
    if not head_file.exists():
        raise FileNotFoundError(f"Head weights not found: {head_file}")
    if not (head_path / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in head directory: {head_path}")

    print(f"Loading backbone: {backbone_file}")
    backbone_tensors = load_safetensors(backbone_file)

    print(f"Loading head: {head_file}")
    head_tensors = load_safetensors(head_file)

    merged = {**backbone_tensors, **head_tensors}
    print(f"Merged {len(backbone_tensors)} backbone + {len(head_tensors)} head tensors → {len(merged)} total")

    output_path.mkdir(parents=True, exist_ok=True)
    merged_weights_path = output_path / "model.safetensors"
    save_file(merged, merged_weights_path)
    size_mb = merged_weights_path.stat().st_size / 1e6
    print(f"Saved merged weights → {merged_weights_path}  ({size_mb:.1f} MB)")

    for filename in INFERENCE_FILES:
        if filename == "model.safetensors":
            continue
        src = head_path / filename
        if src.exists():
            shutil.copy2(src, output_path / filename)
            print(f"Copied {filename}")
        else:
            print(f"  (skipped {filename} — not present in head directory)")

    print(f"\nFull model written to: {output_path}")
    print("Load with: AutoModelForSequenceClassification.from_pretrained(output_path)")


@click.command()
@click.option(
    "--backbone-path",
    type=click.Path(path_type=Path),
    default=Path("models/backbone/backbone.safetensors"),
    show_default=True,
    help="Path to the shared backbone.safetensors file.",
)
@click.option(
    "--head-path",
    type=click.Path(path_type=Path),
    default=Path("models/en_main_head"),
    show_default=True,
    help="Path to the head directory (contains model.safetensors, config.json, tokenizer files).",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("models/coupled"),
    show_default=True,
    help="Directory to write the merged full model into.",
)
def main(backbone_path: Path, head_path: Path, output_path: Path) -> None:
    """Combine backbone and head weights into a single full BERT model."""
    couple(backbone_path, head_path, output_path)


if __name__ == "__main__":
    main()
