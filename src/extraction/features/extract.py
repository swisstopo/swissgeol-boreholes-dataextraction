"""Contains the main extraction pipeline for stratigraphy."""

import logging

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
        filtered_pairs = self._extract_filtered_sidebar_pairs(include_descriptions_without_sidebar=True)

        # Step 4: Create boreholes
        boreholes = [self._create_borehole_from_pair(pair) for pair in filtered_pairs]

        logger.debug(
            f"Page {self.page_number}: Extracted {len(boreholes)} boreholes from {len(self.table_structures)} tables"
        )
        return [
            borehole for borehole in boreholes if len(borehole.predictions) >= self.matching_params["min_num_layers"]
        ]

    def _filter_by_table_criteria(
        self, pairs: list[MaterialDescriptionRectWithSidebar]
    ) -> list[MaterialDescriptionRectWithSidebar]:
        """Filter pairs based on table containment and width requirements."""
        if not self.table_structures:
            return pairs

        filtered = []
        for pair in pairs:
            table_index = self._contained_in_table_index(pair, self.table_structures, proximity_buffer=50)

            # If not in table - keep it as is
            if table_index == -1:
                filtered.append(pair)
            # If in table - check width requirement
            else:
                min_width = self.matching_params["material_description_column_width"] * self.page_width
                if pair.material_description_rect.width > min_width:
                    filtered.append(pair)

        return filtered

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

    def _create_borehole_from_pair(self, pair: MaterialDescriptionRectWithSidebar) -> ExtractedBorehole:
        """Create an ExtractedBorehole from a MaterialDescriptionRectWithSidebar."""
        bounding_boxes = PageBoundingBoxes.from_material_description_rect_with_sidebar(pair, self.page_number)

        interval_block_pairs = self._get_interval_block_pairs(pair)

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
        return ExtractedBorehole(borehole_layers, [bounding_boxes])  # takes a list of bounding boxes

    def _get_interval_block_pairs(self, pair: MaterialDescriptionRectWithSidebar) -> list[IntervalBlockPair]:
        """Get the interval block pairs for a given material description rect with sidebar.

        Args:
            pair (MaterialDescriptionRectWithSidebar): The material description rect with sidebar.

        Returns:
            list[IntervalBlockPair]: The interval block pairs.
        """
        description_lines = get_description_lines(self.lines, pair.material_description_rect)
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

        sidebars_noise.extend(
            AAboveBSidebarExtractor.find_in_words(
                words, line_rtree, list(used_entry_rects), sidebar_params=self.matching_params["depth_column_params"]
            )
        )

        # assign all sidebar to their best match
        material_descriptions_sidebar_pairs = self._match_sidebars_to_description_rects(sidebars_noise)

        return material_descriptions_sidebar_pairs

    def _find_all_material_description_candidates(self, sidebar: Sidebar | None) -> list[pymupdf.Rect]:
        """Find all material description candidates on the page.

        Args:
            sidebar (Sidebar | None): The sidebar for which we want to find the material descriptions.

        Returns:
            list[pymupdf.Rect]: A list of candidate rectangles for material descriptions.
        """
        if sidebar:
            above_sidebar = [
                line for line in self.lines if x_overlap(line.rect, sidebar.rect) and line.rect.y0 < sidebar.rect.y0
            ]

            min_y0 = max(line.rect.y0 for line in above_sidebar) if above_sidebar else -1

            def check_y0_condition(y0):
                return y0 > min_y0 and y0 < sidebar.rect.y1
        else:

            def check_y0_condition(y0):
                return True

        candidate_description = [line for line in self.lines if check_y0_condition(line.rect.y0)]

        is_not_description = [
            line
            for line in candidate_description
            if line.is_description(self.matching_params, self.language, self.analytics, search_excluding=True)
        ]
        is_description = [
            line
            for line in candidate_description
            if line.is_description(self.matching_params, self.language, self.analytics, search_excluding=False)
            and line not in is_not_description
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
                if line.rect.y0 >= best_y0 and line.rect.y1 <= best_y1
                if min_description_x0 < line.rect.x0 < max_description_x0
            ]
            best_x0 = min([line.rect.x0 for line in good_lines])
            best_x1 = max([line.rect.x1 for line in good_lines])

            # check that no lines that have excluded words are contained in the rect
            cluster_rect = pymupdf.Rect(best_x0, best_y0, best_x1, best_y1)
            non_description_in_rect = [
                excl_line
                for excl_line in is_not_description
                if x_overlap_significant_smallest(excl_line.rect, cluster_rect, 0.5)
                and best_y0 < excl_line.rect.y0
                and excl_line.rect.y1 < best_y1
            ]

            # the rect is valid only when description lines are clearly more numerous than non-description lines.
            if len(non_description_in_rect) / len(good_lines) > self.matching_params["non_description_lines_ratio"]:
                continue

            # expand to include entire last block
            def is_below(best_x0, best_y1, line: TextLine):
                return (
                    (line.rect.x0 > best_x0 - 5)
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
                key=lambda rect: MaterialDescriptionRectWithSidebar(sidebar, rect, self.lines).score_match,
            )
        else:
            return candidate_rects[0]

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

    def _extract_filtered_sidebar_pairs(
        self, include_descriptions_without_sidebar: bool = True
    ) -> list[MaterialDescriptionRectWithSidebar]:
        """Extract and filter sidebar pairs using the common pipeline.

        Args:
            include_descriptions_without_sidebar: If True, search for and include
                material descriptions that don't have an associated sidebar.

        Returns:
            List of filtered MaterialDescriptionRectWithSidebar pairs, sorted by
            score (highest first) and filtered by score, table criteria, and
            intersections.
        """
        # Step 1: Find all potential pairs
        pairs = self._find_layer_identifier_sidebar_pairs()
        pairs.extend(self._find_depth_sidebar_pairs())

        # Step 2: Optionally add descriptions without sidebar
        if include_descriptions_without_sidebar:
            # only allow sidebar=None fallback if strong evidence exists
            if self._allow_description_only_fallback():
                material_description_rect_without_sidebar = self._find_material_description_column(sidebar=None)
                if material_description_rect_without_sidebar:
                    pairs.append(
                        MaterialDescriptionRectWithSidebar(
                            sidebar=None,
                            material_description_rect=material_description_rect_without_sidebar,
                            lines=self.lines,
                        )
                    )
            else:
                logger.debug(
                    "Page %s: skipping description-only fallback (insufficient evidence)",
                    self.page_number,
                )

        # Step 3: Sort once by score (highest first)
        pairs.sort(key=lambda pair: pair.score_match, reverse=True)

        # Step 4: Apply filter chain
        filtered_pairs = [pair for pair in pairs if pair.score_match >= 0]
        filtered_pairs = self._filter_by_table_criteria(filtered_pairs)
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
        good_sidebar_pairs = self._extract_filtered_sidebar_pairs(include_descriptions_without_sidebar=False)
        best_sidebar_score = max(
            (pair.score_match for pair in good_sidebar_pairs if pair.sidebar is not None), default=0.0
        )

        return SidebarQualityMetrics(
            number_of_good_sidebars=len(good_sidebar_pairs),
            best_sidebar_score=best_sidebar_score,
        )

    def _allow_description_only_fallback(self) -> bool:
        """Return True if we have strong evidence that a description-only borehole is plausible.

        This is meant to reduce false-positive boreholes created from random paragraphs.
        """
        require_table = self.matching_params.get("fallback_require_table", True)
        allow_if_striplog = self.matching_params.get("fallback_allow_if_striplog", True)

        # Table evidence thresholds
        min_table_height_ratio = self.matching_params.get("fallback_min_table_height_ratio", 0.25)

        has_table = bool(self.table_structures)
        has_striplog = bool(self.strip_logs)

        # 1) If strip-log exists, that's a borehole
        if allow_if_striplog and has_striplog:
            return True

        # 2) Otherwise: require some table evidence
        if require_table and not has_table:
            return False
        if not has_table:
            # no table, no striplog -> don't allow description-only fallback
            return False

        # 3) Table must be "substantial" (avoid tiny tables)
        # If multiple tables: take the largest by height
        largest_table = max(self.table_structures, key=lambda t: t.bounding_rect.height)
        if (largest_table.bounding_rect.height / self.page_height) < min_table_height_ratio:
            return False


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
