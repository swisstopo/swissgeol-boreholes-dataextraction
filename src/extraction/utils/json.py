"""JSON serialization and deserialization logic."""

from decimal import Decimal
from typing import Annotated, Any

import pymupdf
from pydantic import PlainSerializer, PlainValidator


def serialize_rect(rect: pymupdf.Rect) -> list[float]:
    """Serializer for pymupdf.Rect."""
    return [rect.x0, rect.y0, rect.x1, rect.y1]


def deserialize_rect(value: Any) -> pymupdf.Rect:
    """Deserializer for pymupdf.Rect."""
    # 1. If it's already an instantiated Rect, pass it through directly
    if isinstance(value, pymupdf.Rect):
        return value

    # 2. Otherwise, treat it as serialized data
    if isinstance(value, list | tuple) and len(value) == 4:
        return pymupdf.Rect(*value)
    raise ValueError("Invalid format for pymupdf.Rect instantiation")


JsonSerializableRect = Annotated[
    pymupdf.Rect, PlainValidator(deserialize_rect), PlainSerializer(serialize_rect, return_type=list[float])
]


JsonFloatDecimal = Annotated[Decimal, PlainSerializer(lambda x: float(x), return_type=float, when_used="json")]
