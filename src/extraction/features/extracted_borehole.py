"""Module containing a dataclasses for full data on extracted ."""

from dataclasses import dataclass, field

from extraction.features.groundwater.groundwater import Groundwater
from extraction.features.metadata.metadata import BoreholeMetadata
from extraction.features.stratigraphy.layer.layer import Layer
from extraction.features.stratigraphy.layer.page_bounding_boxes import PageBoundingBoxes
from swissgeol_doc_processing.utils.data_extractor import FeatureOnPage


@dataclass
class ExtractedBorehole:
    """A class to store the extracted information of one single borehole."""

    predictions: list[Layer]
    bounding_boxes: list[PageBoundingBoxes]  # one for each page that the borehole spans
    # name/elevation/coordinates matched to this borehole on the page(s) it was (re-)detected on; carried
    # forward across continuation merges (see `_merge_boreholes`)
    metadata: BoreholeMetadata | None = None
    # many-to-one, unlike metadata above: a borehole can have several groundwater readings
    groundwater: list[FeatureOnPage[Groundwater]] = field(default_factory=list)
