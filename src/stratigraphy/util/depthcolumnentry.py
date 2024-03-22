"""Contains dataclasses for entries in a depth column."""

import fitz


class DepthColumnEntry:  # noqa: D101
    def __init__(self, rect: fitz.Rect, value: float):
        self.rect = rect
        self.value = value

    def __repr__(self):
        return str(self.value)


class LayerDepthColumnEntry:  # noqa: D101
    def __init__(self, start: DepthColumnEntry, end: DepthColumnEntry):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"{self.start.value}-{self.end.value}"

    @property
    def rect(self):
        return fitz.Rect(self.start.rect).include_rect(self.end.rect)
