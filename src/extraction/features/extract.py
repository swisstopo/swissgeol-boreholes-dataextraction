"""Contains the main extraction pipeline for stratigraphy."""

import logging
import re

import fastquadtree
import pymupdf

from extraction.features.stratigraphy.interval.interval import IntervalBlockPair
from extraction.features.stratigraphy.layer.layer import (
    ExtractedBorehole,
    Layer,
    LayerDepths,
)
from extraction.features.stratigraphy.layer.page_bounding_boxes import (
    MaterialDescriptionRectWithSidebar,
    PageBoundingBoxes,
)
from extraction.features.stratigraphy.sidebar.classes.sidebar import (
    Sidebar,
    SidebarNoise,
    SidebarQualityMetrics,
    noise_count,
)
from extraction.features.stratigraphy.sidebar.extractor.a_above_b_sidebar_extractor import (
    AAboveBSidebarExtractor,
)
from extraction.features.stratigraphy.sidebar.extractor.a_to_b_sidebar_extractor import (
    AToBSidebarExtractor,
)
from extraction.features.stratigraphy.sidebar.extractor.layer_identifier_sidebar_extractor import (
    LayerIdentifierSidebarExtractor,
)
from extraction.features.stratigraphy.sidebar.extractor.protocol_sidebar_extractor import (
    ProtocolSidebarExtractor,
)
from extraction.features.stratigraphy.sidebar.extractor.spulprobe_sidebar_extractor import SpulprobeSidebarExtractor
from extraction.utils.dynamic_matching import IntervalToLinesDP
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.geometry.line_detection import find_diags_ending_in_zone
from swissgeol_doc_processing.geometry.util import x_overlap, x_overlap_significant_smallest
from swissgeol_doc_processing.text.find_description import get_description_lines
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics
from swissgeol_doc_processing.text.textblock import (
    MaterialDescription,
    MaterialDescriptionLine,
    TextBlock,
)
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.text.textline_affinity import Affinity, get_line_affinity
from swissgeol_doc_processing.utils.data_extractor import FeatureOnPage
from swissgeol_doc_processing.utils.strip_log_detection import StripLog
from swissgeol_doc_processing.utils.table_detection import (
    TableStructure,
)

logger = logging.getLogger(__name__)


class MaterialDescriptionRectWithSidebarExtractor:
    """Class with methods to extract pairs of a material description rect with a corresponding sidebar."""

    def __init__(
        self,
        lines: list[TextLine],
        long_or_horizontal_lines: list[Line],
        all_geometric_lines: list[Line],
        table_structures: list[TableStructure],
        strip_logs: list[StripLog],
        language: str,
        page_number: int,
        page_width: float,
        page_height: float,
        line_detection_params: dict,
        analytics: MatchingParamsAnalytics = None,
        **matching_params: dict,
    ):
        """Creates a new MaterialDescriptionRectWithSidebarExtractor.

        Args:
            lines (list[TextLine]): all the text lines on the page.
            long_or_horizontal_lines (list[Line]): Geometric lines (only the significant ones and the horizontals).
            all_geometric_lines (list[Line]): All the geometric lines of the page (small ones included).
            table_structures (list[TableStructure]): The identified table structures of the page.
            strip_logs (list[StripLog]): The identified strip logs of the page.
            language (str): The language of the page.
            page_number (int): The page number.
            page_width (float): The width of the page.
            page_height (float): The height of the page.
            line_detection_params (dict): The parameters for line detection.
            analytics (MatchingParamsAnalytics): The analytics tracker for matching parameters.
            **matching_params (dict): Additional parameters for the matching pipeline.
        """
        self.lines = lines
        self.long_or_horizontal_lines = long_or_horizontal_lines
        self.all_geometric_lines = all_geometric_lines
        self.table_structures = table_structures
        self.strip_logs = strip_logs  # added for future usage
        self.language = language
        self.page_number = page_number
        self.page_width = page_width
        self.page_height = page_height
        self.line_detection_params = line_detection_params
        self.analytics = analytics
        self.matching_params = matching_params

    def process_page(self) -> list[ExtractedBorehole]:
        """Process a single page of a pdf.

        Finds all descriptions and depth intervals on the page and matches them.

        Returns:
            list[ExtractedBorehole]: The extracted boreholes from the page.
        """
        filtered_pairs = self._extract_filtered_sidebar_pairs()

        valid_boreholes = []
        pairs_from_valid_boreholes = []
        for pair in filtered_pairs:
            borehole = self._create_borehole_from_pair(pair)
            if borehole is not None:
                valid_boreholes.append(borehole)
                pairs_from_valid_boreholes.append(pair)

        material_descriptions_without_sidebar = self._extract_material_descriptions_without_sidebar()
        if material_descriptions_without_sidebar and not any(
            self._pairs_intersect(material_descriptions_without_sidebar, other_pair)
            or self._pairs_adjacent(material_descriptions_without_sidebar, other_pair)
            for other_pair in pairs_from_valid_boreholes
        ):
            # add the material descriptions without sidebar if there is no intersection with any of the already
            # constructed valid boreholes
            borehole = self._create_borehole_from_pair(material_descriptions_without_sidebar)
            if borehole is not None:
                valid_boreholes.append(borehole)

        return valid_boreholes

    def _contained_in_table_index(
        self, pair: MaterialDescriptionRectWithSidebar, table_structures: list[TableStructure], proximity_buffer: float
    ) -> int:
        """Returns the index of the first table structure that contains this pair, or -1 if none is found.

        Args:
            pair: MaterialDescriptionRectWithSidebar object
            table_structures: List of table structures
            proximity_buffer: Distance threshold for proximity check

        Returns:
            The index of the first table structure that contains this pair, or -1 if none is found
        """
        material_rect = pair.material_description_rect
        sidebar_rect = pair.sidebar.rect if pair.sidebar else None

        for index, table in enumerate(table_structures):
            # Check if rectangle is within proximity buffer of table
            expanded_table_rect = pymupdf.Rect(
                table.bounding_rect.x0 - proximity_buffer,
                table.bounding_rect.y0 - proximity_buffer,
                table.bounding_rect.x1 + proximity_buffer,
                table.bounding_rect.y1 + proximity_buffer,
            )

            material_rect_inside = expanded_table_rect.contains(material_rect)
            sidebar_rect_inside = expanded_table_rect.contains(sidebar_rect) if sidebar_rect else True

            if material_rect_inside and sidebar_rect_inside:
                return index

        return -1

    def _filter_by_intersections(
        self, pairs: list[MaterialDescriptionRectWithSidebar]
    ) -> list[MaterialDescriptionRectWithSidebar]:
        """Remove pairs that intersect with higher-scoring pairs."""
        kept_pairs = []

        for pair in pairs:
            # Check if this pair intersects with any already-kept (higher-scoring) pair
            intersects = any(self._pairs_intersect(pair, kept_pair) for kept_pair in kept_pairs)

            # Only keep if no conflicts found
            if not intersects:
                kept_pairs.append(pair)

        return kept_pairs

    def _pairs_intersect(self, pair1, pair2) -> bool:
        """Check if two pairs have any intersecting bounding boxes.

        Creates a bounding box around each pair (union of material description rect and sidebar rect,
        if present) and checks for intersection.
        """
        # Create bounding box for pair1, expanding to include sidebar if present
        bbox1 = pair1.material_description_rect
        if pair1.sidebar:
            bbox1 = bbox1 | pair1.sidebar.rect

        # Create bounding box for pair2 , expanding to include sidebar if present
        bbox2 = pair2.material_description_rect
        if pair2.sidebar:
            bbox2 = bbox2 | pair2.sidebar.rect

        return bbox1.intersects(bbox2)

    def _pairs_adjacent(self, pair1, pair2) -> bool:
        """Check if two pairs are horizontally adjacent with vertical overlap.

        Catches cases where a false-positive column sits immediately next to a
        real borehole column without actually overlapping it.
        """
        bbox1 = pair1.material_description_rect
        if pair1.sidebar:
            bbox1 = bbox1 | pair1.sidebar.rect
        bbox2 = pair2.material_description_rect
        if pair2.sidebar:
            bbox2 = bbox2 | pair2.sidebar.rect
        gap_threshold = self.page_width * 0.05
        y_overlap = bbox1.y0 < bbox2.y1 and bbox2.y0 < bbox1.y1
        h_adjacent = abs(bbox1.x1 - bbox2.x0) < gap_threshold or abs(bbox2.x1 - bbox1.x0) < gap_threshold
        return y_overlap and h_adjacent

    def _create_borehole_from_pair(self, pair: MaterialDescriptionRectWithSidebar) -> ExtractedBorehole | None:
        """Create an ExtractedBorehole from a MaterialDescriptionRectWithSidebar."""
        bounding_boxes = PageBoundingBoxes.from_material_description_rect_with_sidebar(pair, self.page_number)

        interval_block_pairs = self._get_interval_block_pairs(pair)
        # For protocol sidebars, every depth entry must have a matched description.
        if (
            pair.sidebar
            and pair.sidebar.kind == "protocol"
            and not all(ibp.block.lines for ibp in interval_block_pairs)
        ):
            return None

        borehole_layers = [
            Layer(
                material_description=MaterialDescription(
                    text=pair.block.text,
                    lines=[
                        FeatureOnPage(
                            feature=MaterialDescriptionLine(text_line.text),
                            rect=text_line.rect,
                            page=text_line.page_number,
                        )
                        for text_line in pair.block.lines
                    ],
                ),
                depths=LayerDepths.from_interval(pair.depth_interval) if pair.depth_interval else None,
            )
            for pair in interval_block_pairs
        ]

        borehole_layers = [layer for layer in borehole_layers if layer.description_nonempty()]

        min_layers = self.matching_params["min_num_layers"]
        if pair.sidebar and pair.sidebar.kind == "protocol":
            min_layers = self.matching_params.get("protocol_min_num_layers", min_layers)
        if len(borehole_layers) < min_layers:
            return None

        return ExtractedBorehole(borehole_layers, [bounding_boxes])  # takes a list of bounding boxes

    def _get_interval_block_pairs(self, pair: MaterialDescriptionRectWithSidebar) -> list[IntervalBlockPair]:
        """Get the interval block pairs for a given material description rect with sidebar.

        Args:
            pair (MaterialDescriptionRectWithSidebar): The material description rect with sidebar.

        Returns:
            list[IntervalBlockPair]: The interval block pairs.
        """
        description_lines = get_description_lines(self.lines, pair.material_description_rect)
        _header_kws = self.matching_params.get("material_description_column_headers", {}).get(self.language, [])
        description_lines = [
            line
            for line in description_lines
            if not line.is_description(self.matching_params, self.language, None, search_excluding=True)
            and not any(kw.lower() in line.text.lower() for kw in _header_kws)
            and self._line_has_real_word(line.text)
        ]
        diagonals = self.get_diagonals_near_textlines(description_lines, self.line_detection_params)

        line_affinities = get_line_affinity(
            description_lines,
            pair.material_description_rect,
            self.all_geometric_lines,
            self.line_detection_params,
            diagonals,
            block_line_ratio=self.matching_params["block_line_ratio"],
            left_line_length_threshold=self.matching_params["left_line_length_threshold"],
        )
        return (
            match_lines_to_interval(pair.sidebar, description_lines, line_affinities, diagonals)
            if pair.sidebar
            else get_pairs_based_on_line_affinity(description_lines, line_affinities, self.matching_params)
        )

    def _find_layer_identifier_sidebar_pairs(self) -> list[MaterialDescriptionRectWithSidebar]:
        layer_identifier_sidebars = LayerIdentifierSidebarExtractor.from_lines(self.lines)
        material_descriptions_sidebar_pairs = []
        for layer_identifier_sidebar in layer_identifier_sidebars:
            material_description_rect = self._find_material_description_column(layer_identifier_sidebar)
            if material_description_rect:
                material_descriptions_sidebar_pairs.append(
                    MaterialDescriptionRectWithSidebar(layer_identifier_sidebar, material_description_rect, self.lines)
                )
        return material_descriptions_sidebar_pairs

    def _has_valid_description_match(self, sidebar_noise: SidebarNoise) -> bool:
        """Return True if the sidebar can form at least one plausible sidebar/description pair.

        Args:
            sidebar_noise (SidebarNoise): The sidebar noise object containing the sidebar and its noise count.

        Returns:
            bool: True if a valid description match is found, False otherwise.
        """
        candidate_rects = self._find_all_material_description_candidates(
            sidebar_noise.sidebar, use_geometric_seeds=False
        )

        for rect in candidate_rects:
            pair = MaterialDescriptionRectWithSidebar(
                sidebar=sidebar_noise.sidebar,
                material_description_rect=rect,
                lines=self.lines,
                noise_count=sidebar_noise.noise_count,
            )
            if pair.score_match >= 0:
                return True

        return False

    def _should_block_protocol_with_a_above_b(
        self,
        a_above_b_sidebars_noise: list[SidebarNoise],
    ) -> bool:
        """Return True if protocol extraction should be skipped because a usable AAboveB exists."""
        return any(self._has_valid_description_match(sidebar_noise) for sidebar_noise in a_above_b_sidebars_noise)

    def _find_depth_sidebar_pairs(self) -> list[MaterialDescriptionRectWithSidebar]:
        if not self.lines:
            return []

        min_x = min([line.rect.x0 for line in self.lines])
        max_x = max([line.rect.x1 for line in self.lines])
        min_y = min([line.rect.y0 for line in self.lines])
        max_y = max([line.rect.y1 for line in self.lines])
        line_rtree = fastquadtree.RectQuadTreeObjects((min_x, min_y, max_x, max_y), capacity=8)
        for line in self.lines:
            line_rtree.insert((line.rect.x0, line.rect.y0, line.rect.x1, line.rect.y1), obj=line)

        words = sorted([word for line in self.lines for word in line.words], key=lambda word: word.rect.y0)

        # create sidebars with noise count
        spulprobe_sidebars = SpulprobeSidebarExtractor.find_in_lines(self.lines)
        sidebars_noise: list[SidebarNoise] = [
            SidebarNoise(sidebar=sidebar, noise_count=noise_count(sidebar, line_rtree))
            for sidebar in spulprobe_sidebars
        ]
        used_entry_rects = {entry.rect for sidebar in spulprobe_sidebars for entry in sidebar.entries}

        a_to_b_sidebars = AToBSidebarExtractor.find_in_words(words)
        sidebars_noise.extend(
            [
                SidebarNoise(sidebar=sidebar, noise_count=noise_count(sidebar, line_rtree))
                for sidebar in a_to_b_sidebars
            ]
        )
        for column in a_to_b_sidebars:
            for entry in column.entries:
                used_entry_rects.add(entry.start.rect)
                used_entry_rects.add(entry.end.rect)

        a_above_b_sidebars_noise = AAboveBSidebarExtractor.find_in_words(
            words,
            line_rtree,
            self.table_structures,
            list(used_entry_rects),
            sidebar_params=self.matching_params["depth_column_params"],
        )

        sidebars_noise.extend(a_above_b_sidebars_noise)

        for sidebar_noise in a_above_b_sidebars_noise:
            for entry in sidebar_noise.sidebar.entries:
                used_entry_rects.add(entry.rect)

        block_protocol = self._should_block_protocol_with_a_above_b(a_above_b_sidebars_noise)

        if not block_protocol:
            protocol_sidebars_noise = ProtocolSidebarExtractor.find_in_words(
                words,
                self.lines,
                line_rtree,
                list(used_entry_rects),
                self.table_structures,
                sidebar_params=self.matching_params["affinity_params"]["protocol"],
            )
            sidebars_noise.extend(protocol_sidebars_noise)

        # assign all sidebar to their best match
        material_descriptions_sidebar_pairs = self._match_sidebars_to_description_rects(sidebars_noise)

        return material_descriptions_sidebar_pairs

    def _has_separating_line(self, y_top: float, y_bottom: float, x0: float, x1: float) -> bool:
        """Return True if a significant horizontal line crosses the x-range between y_top and y_bottom.

        Used to detect fat table/box borders that visually separate a column header from its
        content, so that the header is never pulled into the material description rect.
        """
        slope_tol = self.line_detection_params.get("line_merging_params", {}).get("horizontal_slope_tolerance", 0.1)
        for line in self.long_or_horizontal_lines:
            if not line.is_horizontal(slope_tol):
                continue
            line_y = (line.start.y + line.end.y) / 2
            if y_top < line_y < y_bottom and line.start.x < x1 and line.end.x > x0:
                return True
        return False

    @staticmethod
    def _line_has_real_word(text: str) -> bool:
        """Return True if text contains at least one real word.

        A real word is a contiguous run of 3+ alphabetic characters that does not
        start with a digit and has at least half its letters in lowercase. Requiring
        contiguous letters rejects abbreviations like "m.ü.M." or "Nr." whose scattered
        alpha chars would otherwise satisfy a simple count check.
        """
        for word in re.findall(r"[^\W\d_]{3,}", text, re.UNICODE):
            alpha = [c for c in word if c.isalpha()]
            if len(alpha) >= 3 and sum(1 for c in alpha if c.islower()) / len(alpha) >= 0.5:
                return True
        return False

    def _has_keyword_match(self, rect: pymupdf.Rect) -> int:
        """Return 1 if rect contains any line matching including_expressions, 0 otherwise.

        Binary so that all keyword-qualifying columns score equally on this dimension,
        making width the true differentiator between candidates.
        Uses direct rect containment (not get_description_lines) to avoid the right-side
        cutoff that would penalize wide columns whose keyword lines fall in the outer 40%.
        """
        return int(
            any(
                line.is_description(self.matching_params, self.language, None, search_excluding=False)
                for line in self.lines
                if rect.contains(line.rect)
            )
        )

    def _has_column_header(self, rect: pymupdf.Rect) -> bool:
        """Return True if an excluded line sits immediately above this rect (a column header).

        Column headers like "Beschreibung" or "Bodenart" are a positive signal that the rect
        is the material description column, and mark its upper content boundary.
        """
        header_keywords = self.matching_params.get("material_description_column_headers", {}).get(self.language, [])
        look_above = self.page_height * 0.05
        for line in self.lines:
            if (
                line.rect.x0 < rect.x1
                and line.rect.x1 > rect.x0
                and rect.y0 - look_above <= line.rect.y1 <= rect.y0 + 5
                and any(kw.lower() in line.text.lower() for kw in header_keywords)
            ):
                return True
        return False

    def _get_geometric_description_seeds(
        self, candidate_description: list[TextLine], sidebar: Sidebar | None
    ) -> list[TextLine] | None:
        """Return seed lines for description clustering based on spatial proximity.

        Uses the right edge of the sidebar — extended to any strip log that sits between
        the sidebar and the description column — as an x-anchor. Every candidate line
        whose left edge starts at or to the right of that anchor is treated as a
        potential description line, removing the dependency on inclusion keywords.

        Returns None when no spatial anchor can be derived (no sidebar and no strip
        logs), in which case the caller should fall back to keyword-based seeds.

        Args:
            candidate_description: Lines already filtered to the correct y-range.
            sidebar: The sidebar paired with this description, or None.

        Returns:
            A (possibly empty) list of seed lines, or None if no anchor is available.
        """
        if not sidebar and not self.strip_logs:
            return None

        tolerance = self.page_width * 0.02
        proximity = self.page_width * 0.15

        if sidebar:
            x_right = sidebar.rect.x1
            x_left = sidebar.rect.x0
            # A strip log sitting between the sidebar and the description column shifts the right anchor
            overlapping_strips = [
                sl for sl in self.strip_logs if x_overlap(sl.bbox, sidebar.rect) and sl.bbox.x1 > x_right
            ]
            if overlapping_strips:
                x_right = max(sl.bbox.x1 for sl in overlapping_strips)

            def close_to_anchor(line: TextLine) -> bool:
                right_side = line.rect.x0 >= x_right - tolerance
                left_side = line.rect.x1 <= x_left + tolerance and line.rect.x1 >= x_left - proximity
                return right_side or left_side
        else:
            # No sidebar: only a right anchor from the strip log edge
            x_right = max(sl.bbox.x1 for sl in self.strip_logs)

            def close_to_anchor(line: TextLine) -> bool:
                return line.rect.x0 >= x_right - tolerance

        seeds = [
            line
            for line in candidate_description
            if close_to_anchor(line)
            and line.rect.width >= line.rect.height  # vertically oriented text is not a description
            and self._line_has_real_word(line.text)  # excludes codes, unit symbols, pure numbers
        ]
        return seeds or None

    def _find_all_material_description_candidates(
        self, sidebar: Sidebar | None, use_geometric_seeds: bool = True
    ) -> list[pymupdf.Rect]:
        """Find all material description candidates on the page.

        Args:
            sidebar (Sidebar | None): The sidebar for which we want to find the material descriptions.
            use_geometric_seeds (bool): When True (default), use spatial proximity to the sidebar /
                strip log as the primary seed signal. Pass False for gating checks such as
                _has_valid_description_match, where the question is specifically "does real
                geological keyword content exist here" — geometric seeding would make that check
                trivially true for any text column and cause false protocol-sidebar blocks.

        Returns:
            list[pymupdf.Rect]: A list of candidate rectangles for material descriptions.
        """
        if sidebar:
            above_sidebar = [
                line for line in self.lines if x_overlap(line.rect, sidebar.rect) and line.rect.y0 < sidebar.rect.y0
            ]

            if above_sidebar:
                min_y0 = max(line.rect.y0 for line in above_sidebar)
            else:
                # No header lines found above the sidebar in its x-column. Use the nearest
                # horizontal separator line above the sidebar as a hard upper boundary; fall
                # back to a small geometric margin so the synthetic "0 → first entry" interval
                # is still reachable.
                slope_tol = self.line_detection_params.get("line_merging_params", {}).get(
                    "horizontal_slope_tolerance", 0.1
                )
                lines_above = [
                    ln
                    for ln in self.long_or_horizontal_lines
                    if ln.is_horizontal(slope_tol)
                    and (ln.start.y + ln.end.y) / 2 < sidebar.rect.y0
                    and ln.start.x < sidebar.rect.x1
                    and ln.end.x > sidebar.rect.x0
                ]
                if lines_above:
                    min_y0 = max((ln.start.y + ln.end.y) / 2 for ln in lines_above)
                else:
                    min_y0 = sidebar.rect.y0 - self.page_height * 0.05

            overlapping_strips = [sl for sl in self.strip_logs if x_overlap(sl.bbox, sidebar.rect)]
            if overlapping_strips:
                strip_top = min(sl.bbox.y0 for sl in overlapping_strips)
                min_y0 = min(min_y0, strip_top - 1) if sidebar.rect.y0 < strip_top else strip_top - 1

            # If a table encloses the sidebar, its top edge is a hard upper boundary —
            # description content cannot start above the table the sidebar lives in.
            for table in self.table_structures:
                if table.bounding_rect.contains(sidebar.rect):
                    min_y0 = max(min_y0, table.bounding_rect.y0 - 1)
                    break

            def check_y0_condition(y0):
                return y0 > min_y0 and y0 < sidebar.rect.y1
        else:
            if self.table_structures:
                # No sidebar or strip log available as a spatial anchor. Restrict the description
                # zone to the largest table on the page so that page titles, footers, and other
                # non-table content cannot enter the description zone.
                largest_table = max(self.table_structures, key=lambda t: t.bounding_rect.height)
                _t_y0, _t_y1 = largest_table.bounding_rect.y0, largest_table.bounding_rect.y1

                def check_y0_condition(y0):
                    return _t_y0 <= y0 <= _t_y1
            else:

                def check_y0_condition(y0):
                    return True

        candidate_description = [line for line in self.lines if check_y0_condition(line.rect.y0)]

        header_keywords = self.matching_params.get("material_description_column_headers", {}).get(self.language, [])

        def is_column_header_line(line: TextLine) -> bool:
            text_lower = line.text.lower()
            return any(kw.lower() in text_lower for kw in header_keywords)

        is_not_description = [
            line
            for line in candidate_description
            if line.is_description(self.matching_params, self.language, self.analytics, search_excluding=True)
            or is_column_header_line(line)
        ]

        # Primary: use geometric proximity to the sidebar / strip log as the seed signal.
        # This captures description lines that lack known inclusion keywords (new terminology,
        # OCR variants, continuation lines) whenever a spatial anchor is available.
        # Fallback: keyword-based inclusion detection for pages without sidebar or strip logs,
        # or when geometric seeding is explicitly disabled (e.g. gating checks).
        geometric_seeds = (
            self._get_geometric_description_seeds(candidate_description, sidebar) if use_geometric_seeds else None
        )
        using_geometric_seeds = geometric_seeds is not None
        if using_geometric_seeds:
            is_description = [line for line in geometric_seeds if line not in is_not_description]
        else:
            # No sidebar or strip log available — use all candidate lines as seeds.
            # The column-level keyword gate later ensures at least one inclusion keyword
            # is present in the accepted cluster, so no per-line keyword check is needed.
            is_description = [
                line
                for line in candidate_description
                if line not in is_not_description
                and line.rect.width >= line.rect.height
                and self._line_has_real_word(line.text)
            ]

        if len(candidate_description) == 0:
            return []

        description_clusters: list[list[TextLine]] = []
        while len(is_description) > 0:
            # 0.4 instead of 0.5 slightly improves geoquat/validation/A76.pdf
            coverage_by_generating_line = [
                [other for other in is_description if x_overlap_significant_smallest(line.rect, other.rect, 0.4)]
                for line in is_description
            ]

            def filter_coverage(coverage: list[TextLine]) -> list[TextLine]:
                if coverage:
                    min_x0 = min(line.rect.x0 for line in coverage)
                    max_x1 = max(line.rect.x1 for line in coverage)
                    x0_threshold = max_x1 - 0.4 * (max_x1 - min_x0)
                    return [line for line in coverage if line.rect.x0 < x0_threshold]
                else:
                    return []

            coverage_by_generating_line = [filter_coverage(coverage) for coverage in coverage_by_generating_line]
            max_coverage = max(coverage_by_generating_line, key=len)
            description_clusters.append(max_coverage)
            is_description = [line for line in is_description if line not in max_coverage]

        candidate_rects = []

        for cluster in description_clusters:
            best_y0 = min([line.rect.y0 for line in cluster])
            best_y1 = max([line.rect.y1 for line in cluster])

            min_description_x0 = min([line.rect.x0 - 0.01 * line.rect.width for line in cluster])
            max_description_x0 = max([line.rect.x0 + 0.2 * line.rect.width for line in cluster])
            good_lines = [
                line
                for line in candidate_description
                if line not in is_not_description  # excluded-keyword lines must not expand the rect
                if line.rect.y0 >= best_y0 and line.rect.y1 <= best_y1
                if min_description_x0 < line.rect.x0 < max_description_x0
                if line.rect.width >= line.rect.height  # vertically oriented text is not a description
                if self._line_has_real_word(line.text)  # pure-number lines (coordinates, heights) are not descriptions
            ]
            best_x0 = min([line.rect.x0 for line in good_lines])
            best_x1 = max([line.rect.x1 for line in good_lines])

            # A material description must contain at least one line with real word content.
            # Rejects clusters that are purely numeric or consist only of codes/identifiers,
            # even when a stray unit symbol or letter-bearing code seeded the cluster.
            if not any(self._line_has_real_word(line.text) for line in good_lines):
                continue

            # The column must contain at least one line matching the inclusion keyword list.
            # This is a column-level gate (not per-line), so geometric and table-based detection
            # still drive which lines are seeds — we just require at least one keyword hit anywhere
            # in the cluster to confirm it is a geological description column.
            if not any(
                line.is_description(self.matching_params, self.language, self.analytics, search_excluding=False)
                for line in good_lines
            ):
                continue

            # check that no lines that have excluded words are contained in the rect
            cluster_rect = pymupdf.Rect(best_x0, best_y0, best_x1, best_y1)
            non_description_in_rect = [
                excl_line
                for excl_line in is_not_description
                if x_overlap_significant_smallest(excl_line.rect, cluster_rect, 0.5)
                and best_y0 <= excl_line.rect.y0  # inclusive: also catch excluded lines at the cluster top
                and excl_line.rect.y1 <= best_y1
            ]

            # the rect is valid only when description lines are clearly more numerous than non-description lines.
            if len(non_description_in_rect) / len(good_lines) > self.matching_params["non_description_lines_ratio"]:
                continue

            # Detect the material description column header (Beschreibung, Bodenart, etc.).
            # Anchor to the FIRST KEYWORD-MATCHING line in good_lines, not to initial_best_y0.
            # initial_best_y0 is often inflated by stray seeds (date lines, page headers) that
            # share the same x-column as the real content — anchoring to the first geological
            # keyword line ensures the search window actually reaches "Beschreibung" even when
            # date lines at y=31 pushed initial_best_y0 far above the real table.
            initial_best_y0 = best_y0
            first_keyword_y0 = min(
                (
                    ln.rect.y0
                    for ln in good_lines
                    if ln.is_description(self.matching_params, self.language, None, search_excluding=False)
                ),
                default=initial_best_y0,
            )
            # Search all page lines
            header_y1 = next(
                (
                    excl_line.rect.y1
                    for excl_line in sorted(self.lines, key=lambda ln: -ln.rect.y1)
                    if excl_line.rect.y1 <= first_keyword_y0 + 5
                    and excl_line.rect.x0 < best_x1
                    and excl_line.rect.x1 > best_x0
                    and any(kw.lower() in excl_line.text.lower() for kw in header_keywords)
                ),
                None,
            )
            # Extend header_y1 downward to cover multi-line column names
            # (e.g. "Beschreibung" line 1, "des aufgeschlossenen Bohrgutes" line 2).
            if header_y1 is not None:
                extended = True
                while extended:
                    extended = False
                    for excl_line in is_not_description:
                        if (
                            excl_line.rect.y0 >= header_y1
                            and excl_line.rect.y1 <= initial_best_y0 + 5
                            and excl_line.rect.x0 < best_x1
                            and excl_line.rect.x1 > best_x0
                            and excl_line.rect.y1 > header_y1
                        ):
                            header_y1 = excl_line.rect.y1
                            extended = True

            # Hard upper boundary: only the column name bottom.
            # Everything below header_y1 is accepted; everything above is rejected.
            # Without a column name, cap is_above expansion at 50 px to avoid runaway chaining.
            upward_limit = header_y1 if header_y1 is not None else initial_best_y0 - 50
            if header_y1 is not None:
                best_y0 = max(best_y0, header_y1)

            def is_above(best_x0, best_x1, best_y0, line: TextLine) -> bool:
                return (
                    line.rect.x0 > best_x0 - 5
                    and line.rect.x0 < (best_x0 + best_x1) / 2
                    and line.rect.y1 > best_y0 - 10  # bottom of the line is just above cluster top
                    and line.rect.y0 < best_y0  # line actually starts above current cluster
                    and line not in is_not_description  # never pull in excluded-keyword lines
                    and not self._has_separating_line(line.rect.y0, best_y0, best_x0, best_x1)
                    and line.rect.y0 >= upward_limit  # noqa: B023
                )

            continue_search = True
            while continue_search:
                line = next(
                    (line for line in candidate_description if is_above(best_x0, best_x1, best_y0, line)),
                    None,
                )
                if line:
                    best_x0 = min(best_x0, line.rect.x0)
                    best_x1 = max(best_x1, line.rect.x1)
                    best_y0 = line.rect.y0
                else:
                    continue_search = False

            # expand to include entire last block
            def is_below(best_x0, best_y1, line: TextLine):
                return (
                    line not in is_not_description  # noqa: B023
                    and (line.rect.x0 > best_x0 - 5)
                    and (line.rect.x0 < (best_x0 + best_x1) / 2)  # noqa: B023
                    and (line.rect.y0 < best_y1 + 10)
                    and (line.rect.y1 > best_y1)
                )

            continue_search = True
            while continue_search:
                line = next((line for line in self.lines if is_below(best_x0, best_y1, line)), None)
                if line:
                    best_x0 = min(best_x0, line.rect.x0)
                    best_x1 = max(best_x1, line.rect.x1)
                    best_y1 = line.rect.y1
                else:
                    continue_search = False

            candidate_rects.append(pymupdf.Rect(best_x0, best_y0, best_x1, best_y1))

        # Suppress narrow rects that are immediately adjacent to a wider one — these are
        # secondary columns sitting next to the real
        # description column, not separate boreholes.
        gap_threshold = self.page_width * 0.05
        suppressed = set()
        for i, r1 in enumerate(candidate_rects):
            if i in suppressed:
                continue
            for j, r2 in enumerate(candidate_rects):
                if i == j or j in suppressed:
                    continue
                y_overlap_exists = r1.y0 < r2.y1 and r2.y0 < r1.y1
                horizontally_adjacent = abs(r1.x1 - r2.x0) < gap_threshold or abs(r2.x1 - r1.x0) < gap_threshold
                if y_overlap_exists and horizontally_adjacent:
                    suppressed.add(i if r1.width < r2.width else j)
        candidate_rects = [r for idx, r in enumerate(candidate_rects) if idx not in suppressed]

        if using_geometric_seeds and len(candidate_rects) > 1:
            # Multiple clusters arise when narrow secondary columns (layer codes, measurements)
            # share the seeding x-range with the real description column. Prefer by keyword match
            # count first (more geological content = more likely the real description column),
            # then fall back to width as a tiebreaker.
            candidate_rects = [max(candidate_rects, key=lambda r: (self._has_keyword_match(r), r.width))]

        return candidate_rects

    def _find_material_description_column(self, sidebar: Sidebar | None) -> pymupdf.Rect | None:
        """Find the best material description column for a given depth column.

        Args:
            sidebar (Sidebar | None): The sidebar to be associated with the material descriptions.

        Returns:
            pymupdf.Rect | None: The material description column.
        """
        candidate_rects = self._find_all_material_description_candidates(sidebar)

        if len(candidate_rects) == 0:
            return None
        if sidebar:
            return max(
                candidate_rects,
                key=lambda rect: (
                    self._has_column_header(rect),
                    self._has_keyword_match(rect),
                    MaterialDescriptionRectWithSidebar(sidebar, rect, self.lines).score_match,
                ),
            )
        else:
            return max(
                candidate_rects,
                key=lambda rect: (
                    self._has_column_header(rect),
                    self._has_keyword_match(rect),
                    rect.width,
                ),
            )

    def _match_sidebars_to_description_rects(
        self, sidebars_noise: list[SidebarNoise]
    ) -> list[MaterialDescriptionRectWithSidebar]:
        """Matches sidebar objects to material description rectangles based on score.

        The algorithm performs greedy matching: each sidebar is paired with the material description rectangle that
        yields the highest score. If the top-scoring rectangle is already matched to another sidebar, the next
        best is considered, and so on. If all potential rectangles are taken, the highest-scoring one is still
        assigned as a default (allowing multiple sidebars to share the same rectangle if necessary).

        Parameters:
            sidebars_noise (List[SidebarNoise]): List of sidebar objects to match.

        Returns:
            List[MaterialDescriptionRectWithSidebar]: List of matched pairs.
        """
        # Store all possible matched rect with the associate scores in a dictionary: (s_idx, rect) -> score
        score_map = {
            (s_idx, rect): MaterialDescriptionRectWithSidebar(sn.sidebar, rect, self.lines, sn.noise_count).score_match
            for s_idx, sn in enumerate(sidebars_noise)
            for rect in self._find_all_material_description_candidates(sn.sidebar)
        }

        matched_pairs = []
        used_sidebars_idx = set()
        used_descr_rects = set()

        def is_valid_pair(
            s_idx: int,
            mat_desc_rect: pymupdf.Rect,
            used_sidebars_idx: set[int],
            sidebars_noise: list[SidebarNoise],
            matched_pairs: list[MaterialDescriptionRectWithSidebar],
        ) -> bool:
            """Check if a sidebar index and material description rectangle can form a valid pair.

            A pair is valid if:
            - The sidebar index has not been used yet.
            - The rectangle has not been used yet.
            - The potential pair does not have an already matched sidebar or rectangle between its elements.
            """
            joined_rect = joined_rect = sidebars_noise[s_idx].sidebar.rect | mat_desc_rect

            return (
                s_idx not in used_sidebars_idx  # don't re-match an already matched sidebar
                and all(
                    (joined_rect & (pair.sidebar.rect | pair.material_description_rect)).is_empty
                    for pair in matched_pairs
                )  # don't allow taking the same rect or crossing pairs (pair having another pair element in between)
            )

        # Step 1: Greedy match based on max scores
        while True:
            # Filter available scores
            available_scores = {
                (s_idx, rect): v
                for (s_idx, rect), v in score_map.items()
                if is_valid_pair(s_idx, rect, used_sidebars_idx, sidebars_noise, matched_pairs)
            }
            if not available_scores:
                break

            # Get best available match
            (best_sidebar_idx, best_rect), _ = max(available_scores.items(), key=lambda x: x[1])
            matched_pairs.append(
                MaterialDescriptionRectWithSidebar(
                    sidebars_noise[best_sidebar_idx].sidebar,
                    best_rect,
                    self.lines,
                    sidebars_noise[best_sidebar_idx].noise_count,
                )
            )
            used_sidebars_idx.add(best_sidebar_idx)
            used_descr_rects.add(best_rect)

        # Step 2: Assign remaining sidebars (if any) to best match (reuse descr_rects)
        remaining_sidebars_idx = [s_idx for s_idx in range(len(sidebars_noise)) if s_idx not in used_sidebars_idx]
        for s_idx in remaining_sidebars_idx:
            sidebar = sidebars_noise[s_idx].sidebar
            noise = sidebars_noise[s_idx].noise_count
            mat_rects = self._find_all_material_description_candidates(sidebar)
            if not mat_rects:
                continue
            best_rect = max(
                mat_rects,
                key=lambda rect: MaterialDescriptionRectWithSidebar(sidebar, rect, self.lines, noise).score_match,
            )
            matched_pairs.append(MaterialDescriptionRectWithSidebar(sidebar, best_rect, self.lines, noise))

        return matched_pairs

    def get_diagonals_near_textlines(
        self, description_lines: list[TextLine], line_detection_params: dict
    ) -> list[Line]:
        """Retrieves the diagonal lines that are near description lines.

        Those diagonal lines indicate that the textline should be matched to an interval higher or below, and not the
        one directly in front of it.

        Args:
            description_lines (list[TextLine]): The description lines.
            line_detection_params (dict): The parameters for line detection.

        Returns:
            list[Line]: The diagonal connectors.
        """
        x0s = [line.rect.x0 for line in description_lines]
        min_x0, max_x0 = min(x0s), max(x0s)
        text_heights = [line.rect.height for line in description_lines]
        min_text_height, max_text_height = min(text_heights), max(text_heights)
        min_y0 = min([line.rect.y0 for line in description_lines])
        max_y1 = max([line.rect.y1 for line in description_lines])

        # Zone where we will look for diagonal line ends, between the strip logs and material descriptions.
        search_zone = pymupdf.Rect(min_x0 - max_text_height, min_y0, max_x0 + max_text_height / 3, max_y1)
        if self.strip_logs:
            left_strip_x1s = [sl.bbox.x1 for sl in self.strip_logs if sl.bbox.x0 < search_zone.x0]
            if left_strip_x1s:
                # Shrink left boundary to the rightmost edge of intersecting strips
                search_zone.x0 = max(search_zone.x0, max(left_strip_x1s))

        # Detect and filter potential diagonals
        diagonals = find_diags_ending_in_zone(self.all_geometric_lines, search_zone)
        diagonals = self._filter_diagonals(
            diagonals, description_lines, min_text_height / 2, max_text_height * 3, line_detection_params
        )
        return diagonals

    def _extract_material_descriptions_without_sidebar(self) -> MaterialDescriptionRectWithSidebar | None:
        """Extract material descriptions without a sidebar (if there is strong enough evidence).

        Returns:
            An optional MaterialDescriptionRectWithSidebar object, which will not have a sidebar.
        """
        # only allow sidebar=None fallback if strong evidence exists
        if self._allow_description_only_fallback():
            material_description_rect_without_sidebar = self._find_material_description_column(sidebar=None)
            if material_description_rect_without_sidebar:
                return MaterialDescriptionRectWithSidebar(
                    sidebar=None,
                    material_description_rect=material_description_rect_without_sidebar,
                    lines=self.lines,
                )
        else:
            logger.debug(
                "Page %s: skipping description-only fallback (insufficient evidence)",
                self.page_number,
            )
        return None

    @staticmethod
    def _get_sidebar_first_depth(sidebar: Sidebar) -> float | None:
        """Return the first (shallowest) numeric depth value from a sidebar, or None.

        Handles all sidebar kinds:
        - AAboveBSidebar / ProtocolSidebar / SpulprobeSidebar: entries are DepthColumnEntry,
          first depth is entries[0].value (float).
        - AToBSidebar: entries are AToBInterval, first depth is entries[0].start.value.
        - LayerIdentifierSidebar: entries have string values — returns None.
        """
        if not sidebar.entries:
            return None
        try:
            if sidebar.kind == "a_to_b":
                start = sidebar.entries[0].start
                return start.value if start is not None else None
            else:
                value = sidebar.entries[0].value
                return float(value) if isinstance(value, (int | float)) else None
        except (AttributeError, IndexError):
            return None

    def _completeness_score_bonus(self, sidebar: Sidebar) -> float:
        """Return a bonus score rewarding sidebars that start near ground level and have many entries.

        A sidebar whose first depth is ≤ 1m is very likely the real borehole profile
        starting from the surface, rather than a sub-interval list (e.g. elevation-based
        entries like 526.40m). The entry-count bonus rewards completeness: a sidebar
        covering many layers is more likely to be the primary description column.
        """
        first_depth = self._get_sidebar_first_depth(sidebar)
        bonus = 0.0
        if first_depth is not None and first_depth <= 1.0:
            bonus += 20.0
        bonus += len(sidebar.entries)
        return bonus

    def _extract_filtered_sidebar_pairs(self) -> list[MaterialDescriptionRectWithSidebar]:
        """Extract and filter sidebar pairs using the common pipeline.

        Returns:
            List of filtered MaterialDescriptionRectWithSidebar pairs, sorted by
            score (highest first) and filtered by score, table criteria, and
            intersections.
        """
        # Step 1: Find all potential pairs
        pairs = self._find_layer_identifier_sidebar_pairs()
        pairs.extend(self._find_depth_sidebar_pairs())

        # Step 2: Hard-filter sidebars whose first depth exceeds the configured maximum.
        # This rejects elevation-based false positives (e.g. 526.40m, 523.50m) that
        # can score well on geometry alone when the real borehole starts near 0m.
        max_first_depth = self.matching_params.get("max_sidebar_first_depth", 300)
        pairs = [
            pair
            for pair in pairs
            if pair.sidebar is None
            or self._get_sidebar_first_depth(pair.sidebar) is None
            or self._get_sidebar_first_depth(pair.sidebar) <= max_first_depth
        ]

        # Step 3: Sort by score + completeness bonus (highest first).
        # The completeness bonus rewards sidebars starting near 0m and having many entries,
        # so that a complete profile (0→20m) is preferred over a sub-interval snippet
        # even when the snippet happens to sit geometrically closer to some text column.
        pairs.sort(
            key=lambda pair: pair.score_match
            + (self._completeness_score_bonus(pair.sidebar) if pair.sidebar is not None else 0),
            reverse=True,
        )

        # Step 4: Apply filter chain
        filtered_pairs = [pair for pair in pairs if pair.score_match >= 0]
        filtered_pairs = self._filter_by_intersections(filtered_pairs)

        return filtered_pairs

    @staticmethod
    def _filter_diagonals(
        g_lines: list[Line],
        description_lines: list[TextLine],
        min_vertical_dist: float,
        max_horizontal_dist: float,
        line_detection_params: dict,
    ) -> list[Line]:
        """Filters the diagonal lines identified."""
        angle_threshold = line_detection_params["diagonals_params"]["angle_threshold"]
        return [
            g_line
            for g_line in g_lines
            if not (
                any(line.rect.contains(g_line.start.tuple) for line in description_lines)
                and any(line.rect.contains(g_line.end.tuple) for line in description_lines)
            )  # lines on text are letters that have segments identified (like W)
            and not g_line.is_vertical(angle_threshold)  # too many other lines are vertical
            and min_vertical_dist < abs(g_line.end.y - g_line.start.y)  # near horizontals are likely noise
            and 0 < g_line.end.x - g_line.start.x < max_horizontal_dist  # lines too long are likely other parasites
        ]

    def extract_sidebars_with_quality_metrics(self) -> SidebarQualityMetrics:
        """Extract all sidebars with quality metrics for classification purposes.

        This method reuses the existing sidebar extraction and matching logic to compute sidebar
        specific metrics

        Returns:
            SidebarQualityMetrics: Quality metrics for all sidebars found on the page.
        """
        # Get filtered pairs (without descriptions without sidebar)
        good_sidebar_pairs = self._extract_filtered_sidebar_pairs()
        best_sidebar_score = max((pair.score_match for pair in good_sidebar_pairs), default=0.0)

        return SidebarQualityMetrics(
            number_of_good_sidebars=len(good_sidebar_pairs),
            best_sidebar_score=best_sidebar_score,
        )

    def _allow_description_only_fallback(self) -> bool:
        """Return True if we have strong evidence that a description-only borehole is plausible.

        This is meant to reduce false-positive boreholes created from random paragraphs.
        """
        # Table evidence thresholds
        min_table_height_ratio = self.matching_params.get("fallback_min_table_height_ratio", 0.85)

        has_table = bool(self.table_structures)
        has_striplog = bool(self.strip_logs)

        # If strip-log exists, that's a borehole
        if has_striplog:
            return True

        if not has_table:
            return False

        # For now, we require the table height to exceed a specific threshold in order to reduce false positives from
        # small tables which might include keywords in their description, misleadingly classifying them as boreholes.
        # BUT: TODO keep in mind that this might exclude some boreholes (e.g scanned image has large margin above and
        # below the actual scanned page) --> this mechanism could/should be optimized in the future!
        largest_table = max(self.table_structures, key=lambda t: t.bounding_rect.height)
        return (largest_table.bounding_rect.height / max(self.page_height, 1e-16)) >= min_table_height_ratio


def extract_page(
    text_lines: list[TextLine],
    long_or_horizontal_lines: list[Line],
    all_geometric_lines: list[Line],
    table_structures: list[TableStructure],
    strip_logs: list[StripLog],
    language: str,
    page_index: int,
    page: pymupdf.Page,
    line_detection_params: dict,
    analytics: MatchingParamsAnalytics,
    **matching_params: dict,
) -> list[ExtractedBorehole]:
    """Process a single PDF page and extract borehole information.

    Acts as a simple interface to MaterialDescriptionRectWithSidebarExtractor without requiring direct class usage.

    Args:
        text_lines (list[TextLine]): All text lines on the page.
        long_or_horizontal_lines (list[Line]): Geometric lines (only the significant ones and the horizontals).
        all_geometric_lines (list[Line]): All geometric lines (including the shorter ones).
        table_structures (list[TableStructure]): The identified table structures.
        strip_logs (list[StripLog]): The identified strip log structures.
        language (str): Language of the page (used in parsing).
        page_index (int): The page index (0-indexed).
        page (pymupdf.Page): The page object from the document.
        line_detection_params (dict): The parameters for line detection.
        analytics (MatchingParamsAnalytics): The analytics tracker for matching parameters.
        **matching_params (dict): Additional parameters for the matching pipeline.

    Returns:
        list[ExtractedBorehole]: Extracted borehole layers from the page.
    """
    # Get page dimensions from the document
    page_width = page.rect.width
    page_height = page.rect.height

    # Extract boreholes
    return MaterialDescriptionRectWithSidebarExtractor(
        text_lines,
        long_or_horizontal_lines,
        all_geometric_lines,
        table_structures,
        strip_logs,
        language,
        page_index + 1,
        page_width,
        page_height,
        line_detection_params,
        analytics,
        **matching_params,
    ).process_page()


def extract_sidebar_information(
    text_lines: list[TextLine],
    long_or_horizontal_lines: list[Line],
    all_geometric_lines: list[Line],
    table_structures: list[TableStructure],
    strip_logs: list[StripLog],
    language: str,
    page_index: int,
    page: pymupdf.Page,
    line_detection_params: dict,
    analytics: MatchingParamsAnalytics,
    **matching_params: dict,
) -> SidebarQualityMetrics:
    """Extract sidebar information with quality metrics.

    This function serves as a simple interface to extract sidebar information
    and quality metrics without requiring direct class usage.

    Args:
        text_lines (list[TextLine]): All text lines on the page.
        long_or_horizontal_lines (list[Line]): Geometric lines (only the significant ones and the horizontals).
        all_geometric_lines (list[Line]): All geometric lines (including the shorter ones).
        table_structures (list[TableStructure]): The identified table structures.
        strip_logs (list[StripLog]): The identified strip log structures.
        language (str): Language of the page (used in parsing).
        page_index (int): The page index (0-indexed).
        page (pymupdf.Page): The page object from the document.
        line_detection_params (dict): The parameters for line detection.
        analytics (MatchingParamsAnalytics): The analytics tracker for matching parameters.
        **matching_params (dict): Additional parameters for the matching pipeline.

    Returns:
        SidebarQualityMetrics: Quality metrics for all sidebars found on the page.
    """
    # Get page dimensions from the document
    page_width = page.rect.width
    page_height = page.rect.height

    # Extract sidebar information
    return MaterialDescriptionRectWithSidebarExtractor(
        text_lines,
        long_or_horizontal_lines,
        all_geometric_lines,
        table_structures,
        strip_logs,
        language,
        page_index + 1,
        page_width,
        page_height,
        line_detection_params,
        analytics,
        **matching_params,
    ).extract_sidebars_with_quality_metrics()


def match_lines_to_interval(
    sidebar: Sidebar,
    description_lines: list[TextLine],
    affinities: list[Affinity],
    diagonals: list[Line],
) -> list[IntervalBlockPair]:
    """Match the description lines to the pair intervals.

    Args:
        sidebar (Sidebar): The sidebar.
        description_lines (list[TextLine]): The description lines.
        affinities (list[Affinity]): the affinity between each line pair, previously computed.
        diagonals (list[Line]): The diagonal lines linking text lines to intervals.

    Returns:
        list[IntervalBlockPair]: The matched depth intervals and text blocks.
    """
    # shift the entries of the sidebar using the diagonals, only relevant for AAboveBSidebars
    if sidebar.kind == "a_above_b":
        sidebar.compute_entries_shift(diagonals)
        sidebar.prevent_shifts_crossing()

    depth_interval_zones = sidebar.get_interval_zone()

    # affinities can differ depending on sidebar type
    affinity_scores = sidebar.dp_weighted_affinities(affinities)
    dp = IntervalToLinesDP(depth_interval_zones, description_lines, affinity_scores)

    _, mapping = dp.solve(sidebar.dp_scoring_fn)

    return sidebar.post_processing(mapping)


def get_pairs_based_on_line_affinity(
    description_lines: list[TextLine], affinities: list[Affinity], matching_params: dict
) -> list[IntervalBlockPair]:
    """Based on the line affinity, group the description lines into blocks.

    The grouping is done based on the presence of geometric lines, the indentation of lines
    and the vertical spacing between lines.

    Args:
        description_lines (list[TextLine]): The text lines to group into blocks.
        affinities (list[Affinity]): the affinity between each line pair, previously computed.
        matching_params (dict): the matching parameters.

    Returns:
        list[IntervalBlockPair]: A list of objects containing the description lines without any interval.
    """
    pairs = []
    prev_block_idx = 0
    weights = matching_params["affinity_params"]["no_sidebar"]["weights"]
    # The presence of >=3 horiz. lines should tighten the vertical spacing constrain
    threshold = -0.99 if sum(affinity.long_lines_affinity for affinity in affinities) <= -3.0 else 0.0
    for line_idx, affinity in enumerate(affinities):
        # note: the affinity of the first line is always 0.0
        if affinity.weighted_affinity(**weights) < threshold:
            pairs.append(IntervalBlockPair(None, TextBlock(description_lines[prev_block_idx:line_idx])))
            prev_block_idx = line_idx

    pairs.append(IntervalBlockPair(None, TextBlock(description_lines[prev_block_idx:])))
    return pairs
