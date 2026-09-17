"""Module containing a dataclasses for full data on extracted ."""

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

    def filter_groundwater_entries(self):
        """Sets the depth and elevation of the groundwater entries of this borehole."""
        borehole_terrain_elevation = self.metadata.elevation.feature.elevation if self.metadata.elevation else None
        self.groundwater.filter_entries(borehole_terrain_elevation, self.predictions)
