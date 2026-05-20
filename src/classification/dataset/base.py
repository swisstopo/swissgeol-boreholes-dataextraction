"""TODO."""

import hashlib
from dataclasses import dataclass

from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class DatasetSample:
    """TODO."""

    filename: str
    borehole_index: int
    layer_index: int
    text: str
    labels: list[str]


class ClassificationDataset(Dataset):
    """TODO."""

    def __init__(self, samples: list[DatasetSample], label2idx: dict[str, int]):
        """TODO."""
        self.samples = samples
        self.label2idx = label2idx

    def _to_encoded_one_hot(self, labels: list[str]) -> list[int]:
        """TODO."""
        encoded = [0] * len(self.label2idx)
        for label in labels:
            encoded[self.label2idx[label]] = 1
        return encoded

    def __len__(self) -> int:
        """TODO."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, list[int]]:
        """TODO."""
        sample = self.samples[idx]
        return sample.text, self._to_encoded_one_hot(sample.labels)


class TokenizedClassificationDataset(Dataset):
    """TODO."""

    def __init__(
        self,
        dataset: ClassificationDataset,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        """TODO."""
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """TODO."""
        return len(self.dataset.samples)

    def __getitem__(self, idx: int) -> dict:
        """TODO."""
        text, labels = self.dataset[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # TODO: check if onehot is okay
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


def _deterministic_hash_ratio(text: str) -> float:
    """Map a string deterministically to a float in [0, 1).

    This is used to assign files to splits in a reproducible way, based only on
    their filename (or any stable string key).

    Args:
        text: Input string to hash (e.g., a filename).

    Returns:
        A float in the half-open interval [0, 1).
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Use the first 8 bytes (64 bits) to build a stable ratio in [0, 1).
    return int.from_bytes(h[:8], "big") / 2**64


# def split(dataset: ClassificationDataset, rtest: float, rvalid: float):
#     """TODO."""

#     splits = {"train": [], "validation": [], "test": []}
#     for sample in dataset.samples:
#         # Extract filename for hash
#         x_ratio = deterministic_hash_ratio(path.name)
#         if x_ratio < rtest:
#             splits["test"].append(path)
#         elif x_ratio < rtest + rvalid:
#             splits["validation"].append(path)
#         else:
#             splits["train"].append(path)

#     return splits["train"], splits["validation"], splits["test"]
