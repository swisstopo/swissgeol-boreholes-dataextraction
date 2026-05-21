"""TODO."""

from classification.dataset.base import ClassificationDataset, DatasetSample
from classification.dataset.enum import ColorsLabel
from core.ground_truth import GroundTruth, GroundTruthLayer


class ColorClassificationDataset(ClassificationDataset):
    """TODO."""

    @staticmethod
    def _layer_to_label(layer: GroundTruthLayer, use_consolidated: bool = True) -> str:
        """TODO."""
        return layer.consolidated.primary_color if use_consolidated else layer.unconsolidated.primary_color

    @staticmethod
    def _is_layer_valid(layer: GroundTruthLayer, use_consolidated: bool = True) -> bool:
        """TODO."""
        return layer.material_description and (
            (use_consolidated and layer.consolidated and layer.consolidated.primary_color)
            or (not use_consolidated and layer.unconsolidated and layer.unconsolidated.primary_color)
        )

    @classmethod
    def from_ground_truth(cls, ground_truth: GroundTruth, use_consolidated: bool = True) -> "ClassificationDataset":
        """TODO."""
        return cls(
            samples=[
                DatasetSample(
                    filename=filename,
                    borehole_index=borehole_index,
                    layer_index=layer_index,
                    text=layer.material_description,
                    labels=cls._layer_to_label(layer, use_consolidated),
                )
                for filename, boreholes in ground_truth.ground_truth.items()
                for borehole_index, borehole in enumerate(boreholes)
                for layer_index, layer in enumerate(borehole.layers)
                if cls._is_layer_valid(layer, use_consolidated)
            ],
            label2idx={color: i for i, color in enumerate(ColorsLabel)},
        )
