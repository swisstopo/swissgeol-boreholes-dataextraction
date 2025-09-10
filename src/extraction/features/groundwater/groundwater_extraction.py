"""This module contains the GroundwaterLevelExtractor class."""

import datetime
import logging
from dataclasses import dataclass

import pymupdf
from scipy.stats import pearsonr

from extraction.features.groundwater.gw_illustration_template_matching import get_entry_near_symbol
from extraction.features.groundwater.utility import extract_date, extract_depth, extract_elevation
from extraction.features.stratigraphy.layer.layer import ExtractedBorehole, Layer
from extraction.features.utils.data_extractor import (
    DataExtractor,
    ExtractedFeature,
    FeatureOnPage,
)
from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.text.textline import TextLine
from extraction.features.utils.utility import get_lines_near_rect

logger = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%d"
MAX_DEPTH = 200  # Maximum depth of the groundwater in meters - Otherwise, depth might be confused with
# elevation from the extraction algorithm.
# TODO: One could use the depth column to find the maximal depth of the borehole and use this as a threshold.


@dataclass
class Groundwater(ExtractedFeature):
    """Abstract class for Groundwater Information."""

    depth: float  # Depth of the groundwater relative to the surface
    date: datetime.date | None = (
        None  # Date of the groundwater measurement, if several dates
        # are present, the date of the document the last measurement is taken
    )
    elevation: float | None = None  # Elevation of the groundwater relative to the mean sea level

    def __str__(self) -> str:
        """Converts the object to a string.

        Returns:
            str: The object as a string.
        """
        return f"Groundwater(date={self.format_date()}, depth={self.depth}, elevation={self.elevation})"

    @staticmethod
    def from_json_values(depth: float | None, date: str | None, elevation: float | None) -> "Groundwater":
        """Converts the object from a dictionary.

        Args:
            depth (float | None): The depth of the groundwater.
            date (str | None): The measurement date of the groundwater.
            elevation (float | None): The elevation of the groundwater.

        Returns:
            Groundwater: The object created from the dictionary.
        """
        date = datetime.datetime.strptime(date, DATE_FORMAT) if date is not None and date != "" else None
        if date > datetime.datetime.now():
            date = date.replace(year=date.year - 100)
        return Groundwater(depth=depth, date=date, elevation=elevation)

    @classmethod
    def from_json(cls, json: dict) -> "Groundwater":
        """Converts a dictionary to an object.

        Args:
            json (dict): A dictionary representing the groundwater information.

        Returns:
            Groundwater: The groundwater information object.
        """
        return cls.from_json_values(
            depth=json["depth"],
            date=json["date"],
            elevation=json["elevation"],
        )

    def format_date(self) -> str | None:
        """Formats the date of the groundwater measurement.

        Returns:
            str | None: The formatted date of the groundwater measurement.
        """
        if self.date is not None:
            return self.date.strftime(DATE_FORMAT)
        else:
            return None

    def to_json(self) -> dict:
        """Converts the object to a dictionary.

        Returns:
            dict: The object as a dictionary.
        """
        return {
            "date": self.format_date(),
            "depth": self.depth,
            "elevation": self.elevation,
        }

    def infer_infos(self, terrain_elevation: float, layers: list[Layer], feature_rect: pymupdf.Rect):
        """Sets the depth and elevation of the groundwater.

        If both informations is missing, it tries to infer them from the given layers and the feature rectangle.

        Args:
            terrain_elevation (float): The elevation of the terrain at the top of the borehole.
            layers (list[Layer]): The list of layers in the borehole.
            feature_rect (pymupdf.Rect): The bounding box of the groundwater feature.
        """
        if self.depth is None and self.elevation is None:
            return
            # y_value = (feature_rect.y0 + feature_rect.y1) / 2

            # Step 1: Prepare y and depth differences
            y_diffs = []
            depth_diffs = []

            for i, layer_i in enumerate(layers):
                for j, layer_j in enumerate(layers):
                    if j <= i:
                        continue  # skip repeats and self-pairs

                    # y-midpoints
                    y_i = (layer_i.rect.y0 + layer_i.rect.y1) / 2
                    y_j = (layer_j.rect.y0 + layer_j.rect.y1) / 2

                    # known depths
                    d_i = layer_i.depth
                    d_j = layer_j.depth

                    if d_i is not None and d_j is not None:
                        y_diffs.append(abs(y_i - y_j))
                        depth_diffs.append(abs(d_i - d_j))

            # Step 2: Compute correlation
            if len(y_diffs) > 1:  # need at least 2 points
                pearson_corr, pearson_p = pearsonr(y_diffs, depth_diffs)
                print("Pearson correlation:", pearson_corr, "p=", pearson_p)

                # Step 3: Decide if y-values are meaningful
                if abs(pearson_corr) > 0.7:  # threshold for "high enough"
                    print("Y-positions are correlated with depth. Inferring missing depths...")

                    # Determine if axis is inverted
                    invert = pearson_corr < 0

                    # Step 4: Infer missing depths using cross-rule
                    for i, layer in enumerate(layers):
                        if layer.depth is None:
                            # find nearest layers with known depths above and below
                            known_above = None
                            known_below = None

                            y_layer = (layer.rect.y0 + layer.rect.y1) / 2
                            if invert:
                                y_layer = -y_layer  # flip axis if inverted

                            for other_layer in layers:
                                if other_layer.depth is None:
                                    continue
                                y_other = (other_layer.rect.y0 + other_layer.rect.y1) / 2
                                if invert:
                                    y_other = -y_other

                                if y_other <= y_layer:
                                    if (
                                        known_above is None
                                        or y_other > (known_above.rect.y0 + known_above.rect.y1) / 2
                                    ):
                                        known_above = other_layer
                                else:
                                    if (
                                        known_below is None
                                        or y_other < (known_below.rect.y0 + known_below.rect.y1) / 2
                                    ):
                                        known_below = other_layer

                            if known_above and known_below:
                                # linear interpolation
                                y_top = (known_above.rect.y0 + known_above.rect.y1) / 2
                                y_bottom = (known_below.rect.y0 + known_below.rect.y1) / 2
                                if invert:
                                    y_top, y_bottom = -y_top, -y_bottom

                                rel_pos = (y_layer - y_top) / (y_bottom - y_top)
                                layer.depth = known_above.depth + rel_pos * (known_below.depth - known_above.depth)
                                print(f"Inferred depth for layer {i}: {layer.depth}")

                else:
                    print("Y-positions do not correlate with depth. Cannot infer missing depths.")

        if self.depth is None and self.elevation is not None:  # if was not already set
            self.depth = round(terrain_elevation - self.elevation, 2)
        if self.elevation is None and self.depth is not None:  # if was not already set
            self.elevation = round(terrain_elevation - self.depth, 2)
        # sanity checks here to correct mistake (e.g. wrong depth extracted... )
        prec = 0.5
        if round(terrain_elevation - self.elevation, 2) - self.depth > prec:  # account for approx, up to 0.5m
            logger.warning(
                f"Extracted groundwater height informations (depth = {self.depth}, elevation = {self.elevation}) "
                f"do not match the constraint 'terrain_elevation - gw_elevation = gw_depth': {terrain_elevation} "
                f"- {self.elevation} = {round(terrain_elevation - self.elevation, 2)} != {self.depth} ± {prec}"
            )


@dataclass
class GroundwatersInBorehole:
    """Class for extracted groundwater information from a single borehole."""

    groundwater_feature_list: list[FeatureOnPage[Groundwater]]

    def to_json(self) -> list[dict]:
        """Converts the object to a list of dictionaries.

        Returns:
            list[dict]: The object as a list of dictionaries.
        """
        return [entry.to_json() for entry in self.groundwater_feature_list]

    @classmethod
    def from_json(cls, json_object: list[dict]) -> "GroundwatersInBorehole":
        """Extract a GroundwatersInBorehole object from a json dictionary.

        Args:
            json_object (list[dict]): the json object containing the informations of the borehole

        Returns:
            GroundwatersInBorehole: the GroundwatersInBorehole object
        """
        return cls([FeatureOnPage.from_json(gw_data, Groundwater) for gw_data in json_object])

    def infer_infos(self, terrain_elevation: float | None, layers: list[Layer]):
        """Sets the depth and elevation of all groundwater entries.

        Args:
            terrain_elevation (float): The elevation of the terrain at the top of the borehole.
            layers (list[Layer]): The list of layers in the borehole.
        """
        for entry in self.groundwater_feature_list:
            entry.feature.infer_infos(terrain_elevation, layers, entry.rect)


@dataclass
class GroundwaterInDocument:
    """Class for extracted groundwater information from a document."""

    groundwater_feature_list: list[FeatureOnPage[Groundwater]]
    filename: str

    def to_json(self) -> list[dict]:
        """Converts the object to a list of dictionaries.

        Returns:
            list[dict]: The object as a list of dictionaries.
        """
        return [entry.to_json() for entry in self.groundwater_feature_list]


class GroundwaterLevelExtractor(DataExtractor):
    """Extract groundwater informations from a PDF document."""

    feature_name = "groundwater"

    # look for elevation values to the left, right and/or immediately below the key
    search_left_factor: float = 2
    search_right_factor: float = 8
    search_below_factor: float = 2
    search_above_factor: float = 0

    preprocess_replacements = {",": ".", "'": ".", "o": "0", "\n": " ", "ü": "u"}

    def __init__(self, language):
        """Initializes the GroundwaterLevelExtractor object.

        Args:
            language (str): the language of the document.
        """
        super().__init__(language)

    def get_text_lines_near_key(self, lines: list[TextLine], page: int) -> list[list[TextLine]]:
        """Extracts the text lines that are close to an explicit "groundwater" label.

        Also apply some preprocessing to the text of those text lines, to deal with some common (OCR) errors.

        Args:
            lines (list[TextLine]): all the lines of text to search in
            page (int): the page number (1-based) of the PDF document

        Returns:
            list[FeatureOnPage[Groundwater]]: all found groundwater information
        """
        # find the key that indicates the groundwater information
        groundwater_key_lines = self.find_feature_key(lines)
        extracted_lines_list = []

        for groundwater_key_line in groundwater_key_lines:
            key_rect = groundwater_key_line.rect
            groundwater_info_lines = self.get_lines_near_key(lines, groundwater_key_line)

            # sort the lines by their proximity to the key line center, compute the distance to the key line center
            key_center = (key_rect.x0 + key_rect.x1) / 2
            extracted_lines_list.append(
                sorted(groundwater_info_lines, key=lambda line: abs((line.rect.x0 + line.rect.x1) / 2 - key_center))
            )
        return extracted_lines_list

    def get_text_lines_near_symbole(
        self,
        lines: list[TextLine],
        geometric_lines: list[Line],
        extracted_boreholes: list[ExtractedBorehole],
    ) -> list[list[TextLine]]:
        seen_depths = [lay.depths for bh in extracted_boreholes for lay in bh.predictions if lay.depths]
        seen_depths = [d for depth in seen_depths for d in (depth.start, depth.end) if d]

        rects = get_entry_near_symbol(lines, geometric_lines, seen_depths)

        # option draw
        # page = document.load_page(page_number - 1)
        # scaling = 3
        # pix = pymupdf.utils.get_pixmap(page, matrix=pymupdf.Matrix(scaling, scaling), clip=page.rect)
        # img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # if pix.n == 4:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # for pair in pairs:
        #     img_debug = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     l1, l2 = pair
        #     cv2.line(
        #         img_debug,
        #         (int(l1.start.x) * scaling, int(l1.start.y) * scaling),
        #         (int(l1.end.x) * scaling, int(l1.end.y) * scaling),
        #         (0, 255, 0),
        #         2,
        #     )  # green
        #     cv2.line(
        #         img_debug,
        #         (int(l2.start.x) * scaling, int(l2.start.y) * scaling),
        #         (int(l2.end.x) * scaling, int(l2.end.y) * scaling),
        #         (255, 0, 0),
        #         2,
        #     )  # blue
        # cv2.imwrite("debug.png", img_debug)

        groundwater_info_lines = [
            get_lines_near_rect(
                self.search_left_factor,
                self.search_right_factor,
                self.search_above_factor,
                self.search_below_factor,
                lines,
                rect,
            )
            for rect in rects
        ]

        return groundwater_info_lines

    def get_groundwater_infos_from_lines(
        self, lines_list: list[list[TextLine]], page_number: int
    ) -> list[FeatureOnPage[Groundwater]]:
        groundwaters = [self.get_groundwater_info_from_lines(lines, page_number) for lines in lines_list]
        return [gw for gw in groundwaters if gw]

    def get_groundwater_info_from_lines(
        self, lines: list[TextLine], page_number: int
    ) -> FeatureOnPage[Groundwater] | None:
        """Extracts the groundwater information from a list of text lines.

        Args:
            lines (list[TextLine]): the lines of text to extract the groundwater information from
            page_number (int): the page number (1-based) of the PDF document
        Returns:
            FeatureOnPage[Groundwater]: the extracted groundwater information
        """
        date: datetime.date | None = None
        depth: float | None = None
        elevation: float | None = None

        matched_lines_rect = []
        for line in lines:
            text = self.preprocess(line.text)

            extracted_date, extracted_date_str = extract_date(text)
            if extracted_date_str and not date:
                date = extracted_date
                text = text.replace(extracted_date_str, "").strip()
                matched_lines_rect.append(line.rect)
            elif extracted_date_str and date:
                continue  # skip extra dates

            depth_val = extract_depth(text, MAX_DEPTH)
            if depth_val and not depth:
                depth = depth_val
                text = text.replace(str(depth), "").strip()
                matched_lines_rect.append(line.rect)

            elevation_val = extract_elevation(text)
            if elevation_val and not elevation:
                elevation = elevation_val
                matched_lines_rect.append(line.rect)

            if date and depth and elevation:
                break

        if not matched_lines_rect:
            return None

        rect_union = matched_lines_rect[0]
        for rect in matched_lines_rect[1:]:
            rect_union |= rect

        # return anyway, we could infer informations latter
        return FeatureOnPage(
            feature=Groundwater(depth=depth, date=date, elevation=elevation),
            rect=rect_union,
            page=page_number,
        )

    def extract_groundwater(
        self,
        page_number: int,
        lines: list[TextLine],
        geometric_lines: list[Line],
        extracted_boreholes: list[ExtractedBorehole],
    ) -> list[FeatureOnPage[Groundwater]]:
        """Extracts the groundwater information from a borehole profile.

        Processes the borehole profile page by page and tries to find the coordinates in the respective text of the
        page.
        Algorithm description:
            1. if that gives no results, search for coordinates close to an explicit "groundwater" label (e.g. "Gswp")

        Args:
            page_number (int): The page number (1-indexed) of the PDF document.
            lines (list[TextLine]): The lines of text to extract the groundwater information from.
            geometric_lines (list[Line]): The geometric lines on the page.
            extracted_boreholes (list[ExtractedBorehole]): The extracted boreholes from the page.

        Returns:
            list[FeatureOnPage[Groundwater]]: the extracted coordinates (if any)
        """
        grounwater_lines_list = self.get_text_lines_near_key(lines, page_number)

        grounwater_lines_list.extend(self.get_text_lines_near_symbole(lines, geometric_lines, extracted_boreholes))

        # found_groundwater = self.get_groundwater_near_key(lines, page_number)
        found_groundwaters = self.get_groundwater_infos_from_lines(grounwater_lines_list, page_number)

        unique_groundwaters = self.filter_duplicates(found_groundwaters)

        if unique_groundwaters:
            groundwater_output = ", ".join([str(entry.feature) for entry in unique_groundwaters])
            logger.info("Found groundwater information on page %s: %s", page_number, groundwater_output)
            return unique_groundwaters

        logger.info("No groundwater found in this borehole profile.")
        return []

    def filter_duplicates(self, found_groundwaters):
        """Filters out duplicate groundwater features from the list.

        Args:
            found_groundwaters (list[FeatureOnPage[Groundwater]]): The list of found groundwater features.

        Returns:
            list[FeatureOnPage[Groundwater]]: The filtered list of unique groundwater features.
        """
        unique_groundwaters = []
        for gw in found_groundwaters:
            keep = True
            to_remove = []
            for other_gw in unique_groundwaters:
                if gw.rect.intersects(other_gw.rect):
                    if gw.rect.get_area() > other_gw.rect.get_area():
                        to_remove.append(other_gw)  # replace smaller with bigger
                    else:
                        keep = False  # skip gw
            for other_gw in to_remove:
                unique_groundwaters.remove(other_gw)
            if keep:
                unique_groundwaters.append(gw)
        return unique_groundwaters
