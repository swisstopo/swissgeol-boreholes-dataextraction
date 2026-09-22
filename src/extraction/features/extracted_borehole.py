"""Module containing a dataclass with full information on an extracted borehole."""

from dataclasses import dataclass, field

from extraction.features.groundwater.groundwater import GroundwatersInBorehole
from extraction.features.metadata.metadata import BoreholeMetadata
from extraction.features.stratigraphy.layer.layer import Layer
from extraction.features.stratigraphy.layer.page_bounding_boxes import PageBoundingBoxes


@dataclass
class ExtractedBorehole:
    """A class to store the extracted information of one single borehole."""

    predictions: list[Layer]
    bounding_boxes: list[PageBoundingBoxes]  # one for each page that the borehole spans
    # name/elevation/coordinates matched to this borehole on the page(s) it was (re-)detected on; carried
    # forward across continuation merges (see `_merge_boreholes`)
    metadata: BoreholeMetadata = field(default_factory=BoreholeMetadata)
    # many-to-one, unlike metadata above: a borehole can have several groundwater readings
    groundwater: GroundwatersInBorehole = field(default_factory=GroundwatersInBorehole)

    def post_processing(self):
        """Finalize the extracted borehole after extraction and matching is complete."""
        max_line_width = self._reference_line_width
        for layer in self.predictions:
            layer.material_description.insert_line_breaks(max_line_width)

        # Infer missing depths and elevation of groundwater and remove duplicated groundwater
        borehole_terrain_elevation = self.metadata.elevation.feature.elevation if self.metadata.elevation else None
        self.groundwater.filter_entries(borehole_terrain_elevation, self.predictions)

    @property
    def _reference_line_width(self) -> float | None:
        """Return the width of the borehole longest description line.

        `MaterialDescription.insert_line_breaks` uses this as a reference for how long a line can get
        before the layout wraps it. Scoped per borehole (not per file): different boreholes, even across
        pages of the same file, can have differently sized description columns.
        """
        line_widths = [line.rect.width for layer in self.predictions for line in layer.material_description.lines]
        return max(line_widths, default=None)
