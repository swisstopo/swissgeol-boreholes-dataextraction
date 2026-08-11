"""This module provides an efficient spatial index for text lines."""

import fastquadtree
import pymupdf

from swissgeol_doc_processing.text.textline import TextLine


class TextLineRTree:
    """Small wrapper around fastquadtree.RectQuadTreeObjects for text lines collections."""

    def __init__(self, text_lines: list[TextLine]):
        if text_lines:
            min_x = min([line.rect.x0 for line in text_lines])
            max_x = max([line.rect.x1 for line in text_lines])
            min_y = min([line.rect.y0 for line in text_lines])
            max_y = max([line.rect.y1 for line in text_lines])
            bounds = (min_x, min_y, max_x, max_y)
        else:
            bounds = (0, 0, 1, 1)
        self.text_line_rtree = fastquadtree.RectQuadTreeObjects(bounds, capacity=8)
        for line in text_lines:
            self.text_line_rtree.insert((line.rect.x0, line.rect.y0, line.rect.x1, line.rect.y1), obj=line)

    def query(self, rect: pymupdf.Rect) -> list[TextLine]:
        return [item.obj for item in self.text_line_rtree.query((rect.x0, rect.y0, rect.x1, rect.y1))]
