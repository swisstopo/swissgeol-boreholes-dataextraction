"""Contains dataclasses for entries in a depth column."""

import fitz


class DepthColumnEntry:  # noqa: D101
    def __init__(self, rect: fitz.Rect, value: float):
        self.rect = rect
        self.value = value

    def __repr__(self):
        return str(self.value)

    def to_json(self):
        return {
            "value": self.value,
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
        }


class AnnotatedDepthColumnEntry(DepthColumnEntry):  # noqa: D101
    """Class to represent a depth column entry obtained from LabelStudio.

    The annotation process in label studio does not come with rectangles for depth column entries.
    Therefore, we set them to None.
    """

    def __init__(self, value):
        self.value = value
        self.rect = None

    def to_json(self):
        return {
            "value": self.value,
            "rect": None,
        }


class LayerDepthColumnEntry:  # noqa: D101
    def __init__(self, start: DepthColumnEntry, end: DepthColumnEntry):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"{self.start.value}-{self.end.value}"

    @property
    def rect(self):
        return fitz.Rect(self.start.rect).include_rect(self.end.rect)

    def to_json(self):
        return {
            "start": self.start.to_json(),
            "end": self.end.to_json(),
            "rect": [self.rect.x0, self.rect.y0, self.rect.x1, self.rect.y1],
        }
