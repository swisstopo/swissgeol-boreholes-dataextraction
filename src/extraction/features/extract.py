"""Contains the main extraction pipeline for stratigraphy."""

import logging
import re

import fastquadtree
import pymupdf

from extraction.features.stratigraphy.borehole_candidate import BoreholeCandidate
from extraction.features.stratigraphy.depth_description_alignment import match_lines_to_interval
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
from extraction.features.stratigraphy.no_sidebar_description_grouping import get_descriptions_blocks
from extraction.features.stratigraphy.sidebar.classes.protocol_sidebar import ProtocolSidebar
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
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.geometry.line_detection import find_diags_ending_in_zone
from swissgeol_doc_processing.geometry.util import x_overlap, x_overlap_significant_smallest
from swissgeol_doc_processing.text.find_description import get_description_lines
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics
from swissgeol_doc_processing.text.textblock import (
    MaterialDescription,
    MaterialDescriptionLine,
)
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.text.textline_affinity import get_line_affinity
from swissgeol_doc_processing.utils.data_extractor import FeatureOnPage
from swissgeol_doc_processing.utils.strip_log_detection import StripLog
from swissgeol_doc_processing.utils.table_detection import (
    TableStructure,
)

logger = logging.getLogger(__name__)


class BoreholeExtractor:
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
        """Creates a new BoreholeExtractor.

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
        valid_candidates = self._extract_filtered_borehole_candidates()

        candidate_without_sidebar = self._extract_borehole_without_sidebar()
        if candidate_without_sidebar and not any(
            candidate_without_sidebar.bounding_box.intersects(other_candidate.bounding_box)
            for other_candidate in valid_candidates
        ):
            # add the material descriptions without sidebar if there is no intersection with any of the already
            # constructed valid boreholes
            valid_candidates.append(candidate_without_sidebar)

        # Only return layers with nonempty description for the time being, for the sake of consistency.
        # TODO After verifying the impact on benchmarking scores, we should also return layers without description.
        return [
            ExtractedBorehole(
                [layer for layer in candidate.borehole.predictions if layer.description_nonempty()],
                candidate.borehole.bounding_boxes,
            )
            for candidate in valid_candidates
        ]

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

    def _filter_by_intersections(self, candidates: list[BoreholeCandidate]) -> list[BoreholeCandidate]:
        """Remove candidates that intersect with higher-scoring candidates."""
        kept_candidates = []

        for candidate in candidates:
            # Check if this pair intersects with any already-kept (higher-scoring) pair
            intersects = any(
                candidate.bounding_box.intersects(kept_candidate.bounding_box) for kept_candidate in kept_candidates
            )

            # Only keep if no conflicts found
            if not intersects:
                kept_candidates.append(candidate)

        return kept_candidates

    def _create_borehole_from_pair(self, pair: MaterialDescriptionRectWithSidebar) -> ExtractedBorehole | None:
        """Create an ExtractedBorehole from a MaterialDescriptionRectWithSidebar."""
        bounding_boxes = PageBoundingBoxes.from_material_description_rect_with_sidebar(pair, self.page_number)

        interval_block_pairs = self._get_interval_block_pairs(pair)
        # For protocol sidebars, every depth entry must have a matched description.
        if (
            pair.sidebar
            and isinstance(pair.sidebar, ProtocolSidebar)
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

        borehole_layers_with_description = [layer for layer in borehole_layers if layer.description_nonempty()]

        min_layers = self.matching_params["min_num_layers"]
        if pair.sidebar and isinstance(pair.sidebar, ProtocolSidebar):
            min_layers = self.matching_params.get("protocol_min_num_layers", min_layers)
        if len(borehole_layers_with_description) < min_layers:
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

        if pair.sidebar:
            return match_lines_to_interval(
                pair.sidebar,
                description_lines,
                line_affinities,
                diagonals,
                self.matching_params["protocol_great_match_threshold"],
            )
        else:
            no_sidebar_weights = self.matching_params["affinity_params"]["no_sidebar"]["weights"]
            return [
                IntervalBlockPair(depth_interval=None, block=text_block)
                for text_block in get_descriptions_blocks(description_lines, line_affinities, no_sidebar_weights)
            ]

    def _find_layer_identifier_candidates(self) -> list[BoreholeCandidate]:
        layer_identifier_sidebars = LayerIdentifierSidebarExtractor.from_lines(self.lines, self.table_structures)
        candidates = []
        for layer_identifier_sidebar in layer_identifier_sidebars:
            if candidate := self._find_borehole_candidate(layer_identifier_sidebar):
                candidates.append(candidate)
        return candidates

    def _has_valid_description_match(self, sidebar_noise: SidebarNoise) -> bool:
        """Return True if the sidebar can form at least one plausible sidebar/description pair.

        Args:
            sidebar_noise (SidebarNoise): The sidebar noise object containing the sidebar and its noise count.

        Returns:
            bool: True if a valid description match is found, False otherwise.
        """
        candidate_rects = self._find_all_material_description_candidates(sidebar_noise.sidebar)

        for rect in candidate_rects:
            pair = MaterialDescriptionRectWithSidebar(
                sidebar=sidebar_noise.sidebar,
                material_description_rect=rect,
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

    def _find_depth_sidebar_candidates(self) -> list[BoreholeCandidate]:
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
        spulprobe_sidebars = SpulprobeSidebarExtractor.find_in_lines(self.lines, self.table_structures)
        sidebars_noise: list[SidebarNoise] = [
            SidebarNoise(sidebar=sidebar, noise_count=noise_count(sidebar, line_rtree))
            for sidebar in spulprobe_sidebars
        ]
        used_entry_rects = {entry.rect for sidebar in spulprobe_sidebars for entry in sidebar.entries}

        a_to_b_sidebars = AToBSidebarExtractor.find_in_words(words, self.table_structures)
        sidebars_noise.extend(
            [
                SidebarNoise(sidebar=sidebar, noise_count=noise_count(sidebar, line_rtree))
                for sidebar in a_to_b_sidebars
            ]
        )
        for column in a_to_b_sidebars:
            for entry in column.entries:
                used_entry_rects.add(entry.rect)

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
        return self._match_sidebars_to_description_rects(sidebars_noise)

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

        horizontal_text_lines = [
            line
            for line in self.lines
            if line.rect.width > line.rect.height and not re.fullmatch(r"[\d\s.,\-/]+", line.text.strip())
        ]
        candidate_description = [line for line in horizontal_text_lines if check_y0_condition(line.rect.y0)]

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
        sorted_above = sorted(candidate_description, key=lambda c: c.rect.y0, reverse=True)

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
            def is_below(best_x0, best_y1, line: TextLine, x_tolerance: float = 5, line_gap: float = 10):
                return (
                    (line.rect.x0 > best_x0 - x_tolerance)
                    and (line.rect.x0 < (best_x0 + best_x1) / 2)  # noqa: B023
                    and (line.rect.y0 < best_y1 + line_gap)
                    and (line.rect.y1 > best_y1)
                )

            def is_above(best_x0, best_y0, line: TextLine, x_tolerance: float = 5, line_gap: float = 10):
                return (
                    (line.rect.x0 > best_x0 - x_tolerance)
                    and (line.rect.x0 < (best_x0 + best_x1) / 2)  # noqa: B023
                    and (line.rect.y1 > best_y0 - line_gap)
                    and (line.rect.y0 < best_y0)
                )

            continue_search = True
            while continue_search:
                line = next((line for line in horizontal_text_lines if is_below(best_x0, best_y1, line)), None)
                if line:
                    best_x0 = min(best_x0, line.rect.x0)
                    best_x1 = max(best_x1, line.rect.x1)
                    best_y1 = line.rect.y1
                else:
                    continue_search = False

            # Expand upward one line at a time.
            # With sidebar: stop at the topmost entry's y-level (avoids column headers above first depth entry).
            # Without sidebar: stop when candidate has sibling lines outside the column (header row signal).
            min_y0_limit = (
                min(e.rect.y0 for e in sidebar.entries) + 5
                if sidebar is not None and sidebar.entries
                else -float("inf")
            )
            while best_y0 > min_y0_limit:
                next_line = next(
                    (
                        desc_line
                        for desc_line in sorted_above
                        if is_above(best_x0, best_y0, desc_line)
                        and (
                            sidebar is not None
                            or not any(
                                other
                                for other in candidate_description
                                if other is not desc_line
                                and abs(other.rect.y0 - desc_line.rect.y0) < desc_line.rect.height
                                and (other.rect.x1 < best_x0 - 10 or other.rect.x0 > best_x1 + 10)
                            )
                        )
                    ),
                    None,
                )
                if next_line is None:
                    break
                best_x0 = min(best_x0, next_line.rect.x0)
                best_x1 = max(best_x1, next_line.rect.x1)
                best_y0 = next_line.rect.y0

            candidate_rects.append(pymupdf.Rect(best_x0, best_y0, best_x1, best_y1))
        return candidate_rects

    def _find_borehole_candidate(self, sidebar: Sidebar | None) -> BoreholeCandidate | None:
        """Create a borehole candidate by finding the best material description column for a given depth column.

        Args:
            sidebar (Sidebar | None): The sidebar to be associated with the material descriptions.

        Returns:
            BoreholeCandidate | None: The selected borehole candidate (if any).
        """
        candidate_boreholes = []
        for rect in self._find_all_material_description_candidates(sidebar):
            pair = MaterialDescriptionRectWithSidebar(sidebar=sidebar, material_description_rect=rect)
            if borehole := self._create_borehole_from_pair(pair):
                candidate_boreholes.append(BoreholeCandidate.from_pair(borehole, pair))

        if len(candidate_boreholes) == 0:
            return None
        if sidebar:
            return max(candidate_boreholes, key=lambda candidate: candidate.score)
        else:
            return candidate_boreholes[0]

    def _match_sidebars_to_description_rects(self, sidebars_noise: list[SidebarNoise]) -> list[BoreholeCandidate]:
        """Matches sidebar objects to material description rectangles based on score.

        The algorithm performs greedy matching: each sidebar is paired with the material description rectangle that
        yields the highest score. If the top-scoring rectangle is already matched to another sidebar, the next
        best is considered, and so on. If all potential rectangles are taken, the highest-scoring one is still
        assigned as a default (allowing multiple sidebars to share the same rectangle if necessary).

        Parameters:
            sidebars_noise (List[SidebarNoise]): List of sidebar objects to match.

        Returns:
            List[BoreholeCandidate]: List of candidate boreholes constructed from matched pairs.
        """
        sidebar_boreholes: dict[int, list[BoreholeCandidate]] = dict()

        for sidebar_index, sn in enumerate(sidebars_noise):
            candidate_boreholes = []
            for rect in self._find_all_material_description_candidates(sn.sidebar):
                pair = MaterialDescriptionRectWithSidebar(sn.sidebar, rect, sn.noise_count)
                if borehole := self._create_borehole_from_pair(pair):
                    candidate_boreholes.append(BoreholeCandidate.from_pair(borehole, pair))

            sidebar_boreholes[sidebar_index] = candidate_boreholes

        selected_boreholes = []
        used_sidebars_idx = set()

        def no_intersection(candidate: BoreholeCandidate, selected_boreholes: list[BoreholeCandidate]) -> bool:
            """Check if the bounding boxes of the candidate do not intersect with selected candidates."""
            joined_rect = candidate.sidebar.rect | candidate.material_description_rect

            return all(
                (joined_rect & (other_candidate.sidebar.rect | other_candidate.material_description_rect)).is_empty
                for other_candidate in selected_boreholes
            )  # don't allow taking the same rect or crossing pairs (pair having another pair element in between)

        # Step 1: Greedy match based on max scores
        while available_boreholes := [
            (sidebar_index, sidebar_boreholes[sidebar_index][borehole_index])
            for sidebar_index, boreholes_for_sidebar in sidebar_boreholes.items()
            if sidebar_index not in used_sidebars_idx
            for borehole_index, candidate in enumerate(boreholes_for_sidebar)
            if no_intersection(candidate, selected_boreholes)
        ]:
            # Get best available match
            best_sidebar_index, best_borehole = max(available_boreholes, key=lambda pair: pair[1].score)
            selected_boreholes.append(best_borehole)
            used_sidebars_idx.add(best_sidebar_index)

        return selected_boreholes

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

    def _extract_borehole_without_sidebar(self) -> BoreholeCandidate | None:
        """Extract material descriptions without a sidebar (if there is strong enough evidence).

        Returns:
            An optional BoreholeCandidate object, which will not have a sidebar.
        """
        # only allow sidebar=None fallback if strong evidence exists
        if self._allow_description_only_fallback():
            return self._find_borehole_candidate(sidebar=None)
        else:
            logger.debug(
                "Page %s: skipping description-only fallback (insufficient evidence)",
                self.page_number,
            )
        return None

    def _extract_filtered_borehole_candidates(self) -> list[BoreholeCandidate]:
        """Extract and filter borehole candidates using the common pipeline.

        Returns:
            List of filtered BoreholeCandidate objects, sorted by score (highest first) and filtered by score, table
            criteria, and intersections.
        """
        # Step 1: Find all potential pairs
        candidates = self._find_layer_identifier_candidates()
        candidates.extend(self._find_depth_sidebar_candidates())

        # Step 2: Sort once by score (highest first)
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)

        # Step 3: Apply filter chain
        filtered_candidates = [candidate for candidate in candidates if candidate.score >= 0]
        filtered_candidates = self._filter_by_intersections(filtered_candidates)

        return filtered_candidates

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
        good_borehole_candidates = self._extract_filtered_borehole_candidates()
        best_candidate_score = max((candidate.core for candidate in good_borehole_candidates), default=0.0)

        return SidebarQualityMetrics(
            number_of_good_sidebars=len(good_borehole_candidates),
            best_sidebar_score=best_candidate_score,
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
