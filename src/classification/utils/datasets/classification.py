"""Classification dataset module."""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import reduce

from classification.utils.file_utils import read_params
from core.ground_truth import GroundTruthBorehole, GroundTruthLayer, GroundTruthLayerDepth, GroundTruthMetadata
from extraction.features.predictions.file_predictions import FilePredictionsWithMetrics
from swissgeol_doc_processing.utils.language_detection import detect_language_of_text

logger = logging.getLogger(__name__)


classification_params = read_params("classification_params.yml")


class GroundTruthBoreholeWithLanguage(GroundTruthBorehole):
    """Ground truth data with predicted language."""

    language: str

    @classmethod
    def _from_boreholes(cls, boreholes: list[GroundTruthBorehole]) -> list[GroundTruthBoreholeWithLanguage]:
        """Detect the language shared by a list of boreholes and attach it to each entry.

        Args:
            boreholes: Source borehole records whose material descriptions are used for language
                detection.

        Returns:
            A new list where each borehole is extended with the detected ``language`` field.
        """
        language = detect_language_of_text(
            text="".join(
                [
                    layer.material_description
                    for borehole in boreholes
                    for layer in borehole.layers
                    if layer.material_description
                ]
            ),
            default_language=classification_params["default_language"],
            supported_languages=classification_params["supported_language"],
        )
        return [cls.model_validate({**borehole.model_dump(), "language": language}) for borehole in boreholes]

    @classmethod
    def _from_prediction(cls, prediction: FilePredictionsWithMetrics) -> list[GroundTruthBoreholeWithLanguage]:
        """Convert a single file's predictions into a list of language-annotated borehole records.

        Detects the language from all material descriptions in the prediction, then wraps each
        borehole as a ``GroundTruthBoreholeWithLanguage`` with empty depth intervals and metadata.

        Args:
            prediction (FilePredictionsWithMetrics): Extraction predictions for a single file.

        Returns:
            A list of ``GroundTruthBoreholeWithLanguage`` entries, one per borehole in the prediction.
        """
        language = detect_language_of_text(
            text="".join(
                [
                    layer.material_description.text
                    for borehole in prediction.boreholes
                    for layer in borehole.layers_in_borehole.layers
                    if layer.material_description
                ]
            ),
            default_language=classification_params["default_language"],
            supported_languages=classification_params["supported_language"],
        )
        return [
            cls.model_validate(
                {
                    "borehole_index": borehole.borehole_index,
                    "layers": [
                        GroundTruthLayer(
                            material_description=layer.material_description.text,
                            depth_interval=GroundTruthLayerDepth(),
                        )
                        for layer in borehole.layers_in_borehole.layers
                    ],
                    "metadata": GroundTruthMetadata(),
                    "groundwater": [],
                    "language": language,
                }
            )
            for borehole in prediction.boreholes
        ]

    @classmethod
    def from_ground_truth(
        cls, ground_truth: dict[str, list[GroundTruthBorehole]]
    ) -> dict[str, list[GroundTruthBoreholeWithLanguage]]:
        """Apply language detection to each group of boreholes.

        Args:
            ground_truth: Mapping from filename to a list of borehole records.

        Returns:
            A new mapping with the same keys where every borehole is extended with the detected
            ``language`` field.
        """
        return {key: cls._from_boreholes(boreholes) for key, boreholes in ground_truth.items()}

    @classmethod
    def from_predictions(
        cls, predictions: list[FilePredictionsWithMetrics]
    ) -> dict[str, list[GroundTruthBoreholeWithLanguage]]:
        """Convert a list of file predictions into a filename-keyed mapping of language-annotated boreholes.

        Args:
            predictions: Extraction predictions, one entry per file.

        Returns:
            A mapping from filename to a list of ``GroundTruthBoreholeWithLanguage`` entries.
        """
        return {prediction.filename: cls._from_prediction(prediction) for prediction in predictions}


def deterministic_hash_ratio(text: str) -> float:
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


def split_samples(
    data: list[LayerInformation], rval: float = 0.15, rtest: float = 0.15
) -> tuple[list[LayerInformation], list[LayerInformation], list[LayerInformation]]:
    """Split a flat list of LayerInformation entries into train, validation, and test subsets.

    Args:
        data: Flat list of LayerInformation entries to split.
        rval: Fraction of data reserved for validation (default 0.15).
        rtest: Fraction of data reserved for testing (default 0.15).

    Returns:
        A tuple (train, val, test) of LayerInformation lists.
    """
    # Get split into sets.
    split_train, split_val, split_test = [], [], []
    for entry in data:
        # Extract filename for hash
        x_ratio = deterministic_hash_ratio(entry.filename)
        if x_ratio < rtest:
            split_test.append(entry)
        elif x_ratio < rtest + rval:
            split_val.append(entry)
        else:
            split_train.append(entry)

    return split_train, split_val, split_test


@dataclass
class LayerInformation:
    """Class for each layer in the ground truth json file."""

    filename: str
    borehole_index: int
    layer_index: int
    language: str
    material_description: str
    class_system: type[ClassificationSystem]
    ground_truth_class: list[ClassificationSystem.EnumMember] | None
    prediction_class: list[ClassificationSystem.EnumMember] | None
    llm_reasoning: str | None

    @staticmethod
    def _to_names(classes: list[ClassificationSystem.EnumMember] | None) -> list[str] | None:
        """Convert a list of enum members to their name strings, or return None if input is None.

        Args:
            classes: Enum members to convert.

        Returns:
            List of enum member name strings, or None if ``classes`` is None.
        """
        return [class_.name for class_ in classes] if classes is not None else None

    @staticmethod
    def _from_names(
        classes: list[str] | None, classification_system: type[ClassificationSystem]
    ) -> list[ClassificationSystem.EnumMember] | None:
        """Resolve a list of class name strings back to enum members using the given classification system.

        Args:
            classes: Class name strings to resolve. Returns None when empty or None.
            classification_system: The classification system used to map each string via
                ``map_most_similar_class``.

        Returns:
            List of resolved enum members, or None if ``classes`` is falsy.
        """
        return [classification_system.map_most_similar_class(class_) for class_ in classes] if classes else None

    def to_json(self) -> dict[str, str | int | None]:
        """Serialize this layer's fields to a JSON-compatible dictionary.

        Returns:
            A flat dictionary with all layer fields.
        """
        return {
            "filename": self.filename,
            "borehole_index": self.borehole_index,
            "layer_index": self.layer_index,
            "language": self.language,
            "material_description": self.material_description,
            "class_system": self.class_system.get_name() if self.class_system else None,
            "ground_truth_class": self._to_names(self.ground_truth_class),
            "prediction_class": self._to_names(self.prediction_class),
            "llm_reasoning": self.llm_reasoning,
        }

    @classmethod
    def from_json(cls, json: dict, classification_system: type[ClassificationSystem]) -> LayerInformation:
        """Deserialize a LayerInformation from a JSON dictionary.

        Args:
            json: Flat dictionary with the keys expected by ``to_json``.
            classification_system: The classification system used to resolve
                class strings back to enum members.

        Returns:
            A new ``LayerInformation`` instance.
        """
        return cls(
            filename=json["filename"],
            borehole_index=json["borehole_index"],
            layer_index=json["layer_index"],
            language=json["language"],
            material_description=json["material_description"],
            class_system=classification_system,
            ground_truth_class=cls._from_names(json["ground_truth_class"], classification_system),
            prediction_class=cls._from_names(json["prediction_class"], classification_system),
            llm_reasoning=json["llm_reasoning"],
        )


class ClassificationTask(IntEnum):
    """Enum representing the type of classification task (single-label, multi-label, or ranked)."""

    single_label = 0
    multi_label = auto()
    ranked = auto()


class ClassificationSystem(ABC):
    """Abstract base class for classification system.

    This class defines the core structure and methods that all classification systems
    should implement. It defines methods for normalizing input class strings, returning
    the corresponding Enum class, and providing a default value for dummy classification.
    """

    EnumClassType = type[IntEnum]  # Type alias for the class that inherit InEnum (e.g. USCSClasses)
    EnumMember = IntEnum  # Type alias for a member of those class (e.g. USCSClasses.CL_ML)

    @classmethod
    @abstractmethod
    def normalize_class_string(cls, class_str: str) -> str:
        """Normalize input class string."""
        ...

    @classmethod
    @abstractmethod
    def get_enum(cls) -> EnumClassType:
        """Return the Enum type associated with the classification."""
        ...

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the name of the system used as a string."""
        ...

    @classmethod
    @abstractmethod
    def get_layer_ground_truth_keys(cls) -> list[list[str]]:
        """Return a list of keys in the layer dictionary that retrieves the ground truth class string."""
        ...

    @classmethod
    def reduce_group(
        cls,
        keys: list[str],
        layer: GroundTruthLayer,
    ) -> list[ClassificationSystem.EnumMember]:
        """Walk an attribute path on a layer and return the resolved enum members.

        Args:
            keys (list[str]): Ordered attribute names forming the path to the ground truth value.
            layer (GroundTruthLayer): A single layer record from which to extract the label.

        Returns:
            list[ClassificationSystem.EnumMember]: Matched enum members, or an empty list if the
                path is absent or the value is ``None``.
        """
        try:
            label_str = reduce(getattr, keys, layer)
        except AttributeError:
            return []

        if label_str is None:
            return []

        if isinstance(label_str, list):
            return [cls.map_most_similar_class(s) for s in label_str]

        return [cls.map_most_similar_class(label_str)]

    @classmethod
    def reduce_label(
        cls,
        layer: GroundTruthLayer,
    ) -> list[ClassificationSystem.EnumMember] | None:
        """Resolve all ground truth labels for a layer across every key group.

        Args:
            layer (GroundTruthLayer): A single layer record to extract labels from.

        Returns:
            list[ClassificationSystem.EnumMember] | None: Flat list of resolved enum members,
                or ``None`` if the layer has no ground truth for this classification system.
        """
        label_groups = [cls.reduce_group(keys, layer) for keys in cls.get_layer_ground_truth_keys()]
        labels_str = [label for label_group in label_groups for label in label_group]

        return labels_str if labels_str else None

    @classmethod
    def process(
        cls, ground_truth: dict[str, list[GroundTruthBoreholeWithLanguage]], allow_none: bool = False
    ) -> list[LayerInformation]:
        """Extract labelled layers from a ground truth mapping as a flat list of LayerInformation entries.

        Args:
            ground_truth (dict[str, list[GroundTruthBoreholeWithLanguage]]): Mapping from filename to a
                list of language-annotated borehole records.
            allow_none (bool): When True, layers without a ground truth label are included (with
                ``ground_truth_class=None``). When False (default), unlabelled layers are skipped.

        Returns:
            list[LayerInformation]: A list of ``LayerInformation``, one per layer across all boreholes.
        """
        return [
            LayerInformation(
                filename=filename,
                borehole_index=borehole_index,
                layer_index=layer_index,
                language=borehole.language,
                material_description=layer.material_description,
                class_system=cls,
                ground_truth_class=cls.reduce_label(layer),
                prediction_class=None,
                llm_reasoning=None,
            )
            for filename, boreholes in ground_truth.items()
            for borehole_index, borehole in enumerate(boreholes)
            for layer_index, layer in enumerate(borehole.layers)
            if (cls.reduce_label(layer) is not None or allow_none) and layer.material_description is not None
        ]

    @classmethod
    @abstractmethod
    def get_default_class_value(cls) -> EnumMember:
        """Return the default value for the enum class."""
        ...

    @classmethod
    def classification_task(cls) -> ClassificationTask:
        """Return the classification task type for this dataset."""
        return ClassificationTask.single_label

    @classmethod
    def map_most_similar_class(cls, class_str: str) -> EnumMember:
        """Maps a string to the closest matching class enum member.

        If no match is found, returns the system's default class via ``get_default_class_value()``.

        Args:
            class_str (str): The input string to map.

        Returns:
            ClassificationType.EnumMember: The matching enum member, or default if no match is found.
        """
        normalized_str = cls.normalize_class_string(class_str)

        classes_enum = cls.get_enum()
        for class_ in classes_enum:
            if normalized_str == class_.name.lower():
                return class_
        logger.warning(
            f"{class_str} does not have a matching class, mapping it to {cls.get_default_class_value().name} instead."
        )
        return cls.get_default_class_value()
