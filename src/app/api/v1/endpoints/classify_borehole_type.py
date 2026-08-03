"""Endpoint for classifying borehole type directly from an uploaded PDF document using BERT."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from app.common.schemas import BoreholeTypePrediction, ClassifyBoreholeTypeResponse
from extraction.features.classification_text import extract_borehole_texts

if TYPE_CHECKING:
    from classification.models.model import BertModel


def classify_borehole_type(
    data: bytes, filename: str, bert_models: dict[str, BertModel]
) -> ClassifyBoreholeTypeResponse:
    """Classify the borehole type of every borehole detected in an uploaded PDF document.

    This endpoint takes the raw bytes of an uploaded PDF:
    It runs the extraction pipeline to detect each borehole and its page
    span, extracts and filters header-like text scoped to that borehole (matching how the model was
    trained), and runs a single full forward pass through the `borehole_type` model per borehole — no
    shared-embedding shortcut across models, since `borehole_type`'s fine-tuned head/pooler layers are
    specific to this model.

    Args:
        data (bytes): Raw bytes of the uploaded PDF document.
        filename (str): Name of the uploaded file, used as an identifier.
        bert_models: Models loaded at startup via the lifespan, keyed by classification system name.

    Returns:
        ClassifyBoreholeTypeResponse: One predicted borehole type per borehole detected in the document.
    """
    bert_model = bert_models["borehole_type"]

    borehole_texts = extract_borehole_texts(BytesIO(data), filename, header_only=True)

    predictions = [
        BoreholeTypePrediction(
            borehole_index=borehole_text.borehole_index,
            class_name=bert_model.predict_class(borehole_text.text)[0].name,
        )
        for borehole_text in borehole_texts
    ]

    return ClassifyBoreholeTypeResponse(boreholes=predictions)
