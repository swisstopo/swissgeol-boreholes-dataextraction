import fitz

from stratigraphy.util.interval import Interval
from stratigraphy.util.textblock import TextBlock

colors = ["purple", "blue"]


def draw_layer(page, interval: Interval | None, block: TextBlock, index: int, is_correct: bool | None):
    if len(block.lines):
        block_rect = block.rect
        color = colors[index % len(colors)]

        # background color for material description
        for line in [line for line in block.lines]:
            page.draw_rect(
                line.rect * page.derotation_matrix, width=0, fill=fitz.utils.getColor(color), fill_opacity=0.2
            )
            if is_correct is not None:
                correct_color = "green" if is_correct else "red"
                page.draw_line(
                    line.rect.top_left * page.derotation_matrix,
                    line.rect.bottom_left * page.derotation_matrix,
                    color=fitz.utils.getColor(correct_color),
                    width=6,
                    stroke_opacity=0.5,
                )

        if interval:
            # background color for depth interval
            background_rect = interval.background_rect
            if background_rect is not None:
                page.draw_rect(
                    background_rect * page.derotation_matrix,
                    width=0,
                    fill=fitz.utils.getColor(color),
                    fill_opacity=0.2,
                )

            # line from depth interval to material description
            line_anchor = interval.line_anchor
            if line_anchor:
                page.draw_line(
                    line_anchor * page.derotation_matrix,
                    fitz.Point(block_rect.x0, (block_rect.y0 + block_rect.y1) / 2) * page.derotation_matrix,
                    color=fitz.utils.getColor(color),
                )
