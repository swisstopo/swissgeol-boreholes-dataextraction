"""Training data providers for the unified BERT training pipeline.

Each provider wraps a data source and produces tokenized HuggingFace Datasets.
Adding a new classification task means implementing TrainingDataProvider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import datasets as hf_datasets

from classification import DATAPATH
from classification.color.dataset_config import DATASET_REGISTRY
from classification.data_loader.color_data_loader import COLORS, BoreholeDataset
from classification.utils.classification_classes import ClassificationSystem
from classification.utils.data_loader import LayerInformation, prepare_classification_data


@dataclass
class DataSplit:
    """Holds tokenized train, validation, and test HuggingFace datasets."""

    train: hf_datasets.Dataset
    val: hf_datasets.Dataset
    test: hf_datasets.Dataset


class TrainingDataProvider(Protocol):
    """Protocol for training data providers used by BertTrainer."""

    name: str

    def get_labels(self) -> list[str]: ...
    def get_split(self, tokenizer, max_length: int) -> DataSplit: ...


def _tokenize(texts: list[str], labels: list[int], tokenizer, max_length: int) -> hf_datasets.Dataset:
    raw = hf_datasets.Dataset.from_dict({"text": texts, "label": labels})
    return raw.map(
        lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


@dataclass
class ColorDataProvider:
    """Provides train/val/test splits for color classification.

    Conso and unconso data are each split once with a fixed seed. Experiments
    select from these pre-computed splits so the same 80 % of samples is always
    used for training, regardless of which experiment is run.

    Split logic (matches the consolidation study table):
        - val tracks the TRAIN distribution for non-mixed experiments
        - val tracks the TEST distribution for mixed experiments

    Args:
        train_consolidation: 1=conso, 0=unconso, None=mixed (conso+unconso).
        test_consolidation:  1=conso, 0=unconso.
        name: Experiment name used for output directory naming.
    """

    train_consolidation: int | None
    test_consolidation: int
    name: str = "color_experiment"
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def get_labels(self) -> list[str]:
        return COLORS

    def _load_consolidation(self, consolidation: int) -> list:
        samples = []
        for path, cons in DATASET_REGISTRY.values():
            if cons == consolidation:
                samples.extend(BoreholeDataset.from_json(path).filter(consolidated=consolidation))
        return samples

    def get_split(self, tokenizer, max_length: int) -> DataSplit:
        needed = {self.test_consolidation}
        if self.train_consolidation is None:
            needed |= {0, 1}
        else:
            needed.add(self.train_consolidation)

        pre_split = {
            cons: BoreholeDataset.random_split(
                self._load_consolidation(cons), self.val_ratio, self.test_ratio, self.seed
            )
            for cons in needed
        }

        val_cons = self.test_consolidation if self.train_consolidation is None else self.train_consolidation

        train_ds = (
            BoreholeDataset(list(pre_split[0][0]) + list(pre_split[1][0]))
            if self.train_consolidation is None
            else pre_split[self.train_consolidation][0]
        )
        val_ds = pre_split[val_cons][1]
        test_ds = pre_split[self.test_consolidation][2]

        def _to_hf(ds: BoreholeDataset) -> hf_datasets.Dataset:
            texts, labels = [], []
            for sample in ds:
                if not any(sample["color"]):
                    continue
                texts.append(sample["material_description"])
                labels.append(sample["color"].index(1))
            return _tokenize(texts, labels, tokenizer, max_length)

        return DataSplit(train=_to_hf(train_ds), val=_to_hf(val_ds), test=_to_hf(test_ds))


def consolidation_study_providers() -> list[ColorDataProvider]:
    """Return the 6 ColorDataProviders for the consolidation generalisation study.

    conso   → conso    (in-distribution baseline)
    conso   → unconso  (cross-type generalisation)
    unconso → conso    (cross-type generalisation)
    unconso → unconso  (in-distribution baseline)
    mixed   → conso    (does mixed training help on conso?)
    mixed   → unconso  (does mixed training help on unconso?)
    """
    return [
        ColorDataProvider(train_consolidation=1, test_consolidation=1, name="conso_to_conso"),
        ColorDataProvider(train_consolidation=1, test_consolidation=0, name="conso_to_unconso"),
        ColorDataProvider(train_consolidation=0, test_consolidation=1, name="unconso_to_conso"),
        ColorDataProvider(train_consolidation=0, test_consolidation=0, name="unconso_to_unconso"),
        ColorDataProvider(train_consolidation=None, test_consolidation=1, name="mixed_to_conso"),
        ColorDataProvider(train_consolidation=None, test_consolidation=0, name="mixed_to_unconso"),
    ]


@dataclass
class LithologyDataProvider:
    """Wraps the existing lithology/USCS/EN data loading (separate train + eval files).

    For uscs and en_main the same file is used for both train and val.
    There is no held-out test set; val is reused as test for metric reporting.
    """

    config: dict
    classification_system: type[ClassificationSystem]
    name: str = "lithology_experiment"

    def get_labels(self) -> list[str]:
        classes = self.classification_system.get_enum()
        return [c.name for c in sorted(classes, key=lambda c: c.value)]

    def get_split(self, tokenizer, max_length: int) -> DataSplit:
        sys_name = self.classification_system.get_name()
        if sys_name in ("uscs", "en_main"):
            path = DATAPATH / self.config["json_file_name"]
            train_layers = prepare_classification_data(path, None, self.classification_system)
            val_layers = train_layers
        else:
            train_layers = prepare_classification_data(
                DATAPATH / self.config["train_subset"], None, self.classification_system
            )
            val_layers = prepare_classification_data(
                DATAPATH / self.config["eval_subset"], None, self.classification_system
            )

        def _to_hf(layers: list[LayerInformation]) -> hf_datasets.Dataset:
            texts = [layer.material_description for layer in layers]
            labels = [layer.ground_truth_class.value for layer in layers]
            return _tokenize(texts, labels, tokenizer, max_length)

        val_hf = _to_hf(val_layers)
        return DataSplit(train=_to_hf(train_layers), val=val_hf, test=val_hf)
