"""Generic dataloader for BERT fine-tuning on borehole layer data."""

from __future__ import annotations

import csv
import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

from classification import PROJECT_ROOT

logger = logging.getLogger(__name__)

_CODELIST_COLOR_PATH = PROJECT_ROOT / "src/classification/classification_data/codelist_color.csv"

# Language column names in the codelist CSV.
COLOR_LANGUAGES = ("text_cli_en", "text_cli_de", "text_cli_fr", "text_cli_it")

# Sentinel values that represent "no color specified" across languages.
_NO_COLOR_VALUES = {
    "other",
    "not specified",  # en
    "andere",
    "keine Angabe",  # de
    "autre",
    "sans indication",  # fr
    "altro",
    "senza indicazioni",  # it
}


def _load_color_index(path: Path = _CODELIST_COLOR_PATH) -> tuple[list[str], dict[str, int]]:
    """Build the canonical color list and a cross-language lookup from the codelist CSV.

    Returns:
        colors: Ordered list of English color names (one per concept row). This defines
                the one-hot vector length and the meaning of each index.
        color_to_idx: Maps any language variant of a color (lowercased) → its index.
    """
    colors: list[str] = []
    color_to_idx: dict[str, int] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            canonical = row["text_cli_en"]
            if canonical in _NO_COLOR_VALUES:
                continue
            idx = len(colors)
            colors.append(canonical)
            for lang_col in COLOR_LANGUAGES:
                variant = row[lang_col].lower().strip()
                if variant and variant not in _NO_COLOR_VALUES:
                    color_to_idx[variant] = idx
    return colors, color_to_idx


COLORS, _COLOR_TO_IDX = _load_color_index()


class LayerSample(TypedDict):
    """A single borehole layer sample with labels for BERT fine-tuning.

    Fields:
        material_description: Raw text description of the layer material.
        color: One-hot vector over COLORS. All zeros when color is unknown.
        consolidated: 1 = consolidated, 0 = unconsolidated, None = not specified.
    """

    material_description: str
    color: list[int]
    consolidated: int | None


def _encode_color(color: str | None) -> list[int]:
    """Encode a color string as a one-hot vector over COLORS, or all-zeros if unknown.

    Args: color: Color name in any language, or None.

    Returns: One-hot encoding of the color, where the index is determined by COLORS. If the color
    """
    vec = [0] * len(COLORS)
    if color is None:
        return vec
    idx = _COLOR_TO_IDX.get(color.lower().strip())
    if idx is None:
        logger.warning("Unknown color '%s', encoding as all-zeros.", color)
    else:
        vec[idx] = 1
    return vec


def _extract_primary_color(layer: dict) -> str | None:
    """Return the primary_color string from inside consolidated or unconsolidated, or None.

    Returns None for sentinel values (e.g. 'not specified', 'keine Angabe').
    """
    for key in ("consolidated", "unconsolidated"):
        sub = layer.get(key) or {}
        color = sub.get("primary_color")
        if color and color.lower().strip() not in _NO_COLOR_VALUES:
            return color
    return None


def _encode_consolidation(layer: dict) -> int | None:
    if layer.get("consolidated") is not None:
        return 1
    if layer.get("unconsolidated") is not None:
        return 0
    return None


class BoreholeDataset:
    """Iterable dataset of borehole layer samples loaded from a ground truth JSON.

    Each item is a LayerSample dict with keys:
        - material_description (str)
        - color (list[int], one-hot over COLORS)
        - consolidated (int | None)
    """

    def __init__(self, samples: list[LayerSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> LayerSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[LayerSample]:
        return iter(self._samples)

    @classmethod
    def from_json(
        cls,
        path: Path,
        skip_unknown_consolidation: bool = False,
    ) -> BoreholeDataset:
        """Load a BoreholeDataset from a ground truth JSON file.

        Args:
            path: Path to the ground truth JSON (maps filename → list of boreholes).
            skip_unknown_consolidation: When True, layers where neither 'consolidated'
                nor 'unconsolidated' is set are excluded from the dataset.

        Returns:
            BoreholeDataset with one LayerSample per valid layer.
        """
        with open(path, encoding="utf-8") as f:
            ground_truth = json.load(f)

        samples: list[LayerSample] = []
        for filename, boreholes in ground_truth.items():
            for borehole in boreholes:
                for layer in borehole.get("layers", []):
                    material_description = layer.get("material_description")
                    if not material_description:
                        continue

                    consolidated = _encode_consolidation(layer)
                    if skip_unknown_consolidation and consolidated is None:
                        logger.debug("Skipping layer in %s: consolidation type not specified.", filename)
                        continue

                    samples.append(
                        LayerSample(
                            material_description=material_description,
                            color=_encode_color(_extract_primary_color(layer)),
                            consolidated=consolidated,
                        )
                    )

        logger.info("Loaded %d layer samples from %s.", len(samples), path)
        return cls(samples)

    def filter(self, *, consolidated: int | None = None) -> BoreholeDataset:
        """Return a new BoreholeDataset keeping only samples with the given consolidation value.

        Args:
            consolidated: 1 = consolidated only, 0 = unconsolidated only, None = not specified.

        Returns:
            A new BoreholeDataset with the filtered samples.
        """
        return BoreholeDataset([s for s in self._samples if s["consolidated"] == consolidated])

    @classmethod
    def random_split(
        cls,
        samples: list[LayerSample],
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[BoreholeDataset, BoreholeDataset, BoreholeDataset]:
        """Split samples into train/val/test with a fixed random shuffle.

        The split is fully deterministic for a given seed, so the same 80 % of
        samples are always used for training regardless of which experiment calls
        this method.

        Args:
            samples: All samples to split (typically all data for one consolidation type).
            val_ratio: Fraction held out for validation.
            test_ratio: Fraction held out for testing.
            seed: Random seed for reproducibility.

        Returns:
            (train, val, test) BoreholeDatasets with no overlap.
        """
        shuffled = list(samples)
        random.Random(seed).shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        return (
            cls(shuffled[n_test + n_val :]),
            cls(shuffled[n_test : n_test + n_val]),
            cls(shuffled[:n_test]),
        )
