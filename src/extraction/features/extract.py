"""Contains the main extraction pipeline for stratigraphy."""

import logging

import pymupdf
import rtree

from extraction.features.stratigraphy.interval.interval import AAboveBInterval, Interval, IntervalBlockPair
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
from extraction.features.utils.data_extractor import FeatureOnPage
from extraction.features.utils.geometry.geometry_dataclasses import Line
from extraction.features.utils.geometry.util import x_overlap, x_overlap_significant_smallest
from extraction.features.utils.strip_log_detection import StripLog
from extraction.features.utils.table_detection import (
    TableStructure,
    _contained_in_table_index,
)
from extraction.features.utils.text.find_description import (
    get_description_blocks,
    get_description_lines,
)
from extraction.features.utils.text.matching_params_analytics import MatchingParamsAnalytics
from extraction.features.utils.text.textblock import (
    MaterialDescription,
    MaterialDescriptionLine,
    TextBlock,
)
from extraction.features.utils.text.textline import TextLine

logger = logging.getLogger(__name__)


class MaterialDescriptionRectWithSidebarExtractor:
    """Class with methods to extract pairs of a material description rect with a corresponding sidebar."""

    def __init__(
        self,
        lines: list[TextLine],
        geometric_lines: list[Line],
        table_structures: list[TableStructure],
        strip_logs: list[StripLog],
        language: str,
        page_number: int,
        page_width: float,
        page_height: float,
        analytics: MatchingParamsAnalytics = None,
        **params: dict,
    ):
        """Creates a new MaterialDescriptionRectWithSidebarExtractor.

        Args:
            lines (list[TextLine]): all the text lines on the page.
            geometric_lines (list[Line]): The geometric lines of the page.
            table_structures (list[TableStructure]): The identified table structures of the page.
            strip_logs (list[StripLog]): The identified strip logs of the page.
            language (str): The language of the page.
            page_number (int): The page number.
            page_width (float): The width of the page.
            page_height (float): The height of the page.
            analytics (MatchingParamsAnalytics): The analytics tracker for matching parameters.
            **params (dict): Additional parameters for the matching pipeline.
        """
        self.lines = lines
        self.geometric_lines = geometric_lines
        self.table_structures = table_structures
        self.strip_logs = strip_logs  # added for future usage
        self.language = language
        self.page_number = page_number
        self.page_width = page_width
        self.page_height = page_height
        self.analytics = analytics
        self.params = params

    def process_page(self) -> list[ExtractedBorehole]:
        """Process a single page of a pdf.

        Finds all descriptions and depth intervals on the page and matches them.

        Returns:
            list[ExtractedBorehole]: The extracted boreholes from the page.
        """
        # Step 1: Find all potential pairs
        material_descriptions_sidebar_pairs = self._find_layer_identifier_sidebar_pairs()
        material_descriptions_sidebar_pairs.extend(self._find_depth_sidebar_pairs())

        material_description_rect_without_sidebar = self._find_material_description_column(sidebar=None)
        if material_description_rect_without_sidebar:
            material_descriptions_sidebar_pairs.append(
                MaterialDescriptionRectWithSidebar(
                    sidebar=None, material_description_rect=material_description_rect_without_sidebar, lines=self.lines
                )
            )

        # Step 2: Sort once by score (highest first)
        material_descriptions_sidebar_pairs.sort(key=lambda pair: pair.score_match, reverse=True)

        # Step 3: Apply filter chain
        filtered_pairs = [pair for pair in material_descriptions_sidebar_pairs if pair.score_match >= 0]
        filtered_pairs = self._filter_by_table_criteria(filtered_pairs)
        filtered_pairs = self._filter_by_intersections(filtered_pairs)

        # Step 4: Create boreholes
        boreholes = [self._create_borehole_from_pair(pair) for pair in filtered_pairs]

        logger.debug(
            f"Page {self.page_number}: Extracted {len(boreholes)} boreholes from {len(self.table_structures)} tables"
        )
        return [borehole for borehole in boreholes if len(borehole.predictions) >= self.params["min_num_layers"]]

    def _filter_by_table_criteria(
        self, pairs: list[MaterialDescriptionRectWithSidebar]
    ) -> list[MaterialDescriptionRectWithSidebar]:
        """Filter pairs based on table containment and width requirements."""
        if not self.table_structures:
            return pairs

        filtered = []
        for pair in pairs:
            table_index = _contained_in_table_index(pair, self.table_structures, proximity_buffer=50)

            # If not in table - keep it as is
            if table_index == -1:
                filtered.append(pair)
            # If in table - check width requirement
            else:
                min_width = self.params["material_description_column_width"] * self.page_width
                if pair.material_description_rect.width > min_width:
                    filtered.append(pair)

        return filtered

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
        bounding_boxes = PageBoundingBoxes.from_material_description_rect_with_sidebar(pair, self.page_number)

        # this must not be flattened anymore, i.e. we keep the /per borehole separation
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
        if pair.sidebar:
            return _match_columns(
                pair.sidebar,
                description_lines,
                self.geometric_lines,
                pair.material_description_rect,
                **self.params,
            )
        else:
            description_blocks = get_description_blocks(
                description_lines,
                self.geometric_lines,
                pair.material_description_rect,
                self.params["block_line_ratio"],
                self.params["left_line_length_threshold"],
            )
            return [IntervalBlockPair(block=block, depth_interval=None) for block in description_blocks]

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
        line_rtree = rtree.index.Index()
        for line in self.lines:
            line_rtree.insert(id(line), (line.rect.x0, line.rect.y0, line.rect.x1, line.rect.y1), obj=line)

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
                words, line_rtree, list(used_entry_rects), sidebar_params=self.params["depth_column_params"]
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
            if line.is_description(self.params, self.language, self.analytics, search_excluding=True)
        ]
        is_description = [
            line
            for line in candidate_description
            if line.is_description(self.params, self.language, self.analytics, search_excluding=False)
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
            if len(non_description_in_rect) / len(good_lines) > self.params["non_description_lines_ratio"]:
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


def extract_page(
    text_lines: list[TextLine],
    geometric_lines: list[Line],
    table_structures: list[TableStructure],
    strip_logs: list[StripLog],
    language: str,
    page_index: int,
    document: pymupdf.Document,
    analytics: MatchingParamsAnalytics,
    **matching_params: dict,
) -> list[ExtractedBorehole]:
    """Process a single PDF page and extract borehole information.

    Acts as a simple interface to MaterialDescriptionRectWithSidebarExtractor without requiring direct class usage.

    Args:
        text_lines (list[TextLine]): All text lines on the page.
        geometric_lines (list[Line]): Geometric lines (e.g., from layout analysis).
        table_structures (list[TableStructure]): The identified table structures.
        strip_logs (list[StripLog]): The identified strip log structures.
        language (str): Language of the page (used in parsing).
        page_index (int): The page index (0-indexed).
        document (pymupdf.Document): the document.
        analytics (MatchingParamsAnalytics): The analytics tracker for matching parameters.
        **matching_params (dict): Additional parameters for the matching pipeline.

    Returns:
        list[ExtractedBorehole]: Extracted borehole layers from the page.
    """
    # Get page dimensions from the document
    page = document[page_index]
    page_width = page.rect.width
    page_height = page.rect.height

    # Extract boreholes
    return MaterialDescriptionRectWithSidebarExtractor(
        text_lines,
        geometric_lines,
        table_structures,
        strip_logs,
        language,
        page_index + 1,
        page_width,
        page_height,
        analytics,
        **matching_params,
    ).process_page()


def _match_columns(
    sidebar: Sidebar,
    description_lines: list[TextLine],
    geometric_lines: list[Line],
    material_description_rect: pymupdf.Rect,
    **params: dict,
) -> list[IntervalBlockPair]:
    """Match the layers that can be derived from the sidebar with the description lines.

    This function identifies groups of depth intervals and text blocks that are likely to match.
    The actual matching between text blocks and depth intervals is handled by the implementation of the actual Sidebar
    instance (e.b. AAboveBSidebar, AToBSidebar).

    Args:
        sidebar (Sidebar): The sidebar.
        description_lines (list[TextLine]): The description lines.
        geometric_lines (list[Line]): The geometric lines.
        material_description_rect (pymupdf.Rect): The material description rectangle.
        **params (dict): Additional parameters for the matching pipeline.

    Returns:
        list[IntervalBlockPair]: The matched depth intervals and text blocks.
    """
    return [
        element
        for group in sidebar.identify_groups(description_lines, geometric_lines, material_description_rect, **params)
        for element in _transform_groups(group.depth_intervals, group.blocks)
    ]


def _transform_groups(depth_intervals: list[Interval], blocks: list[TextBlock]) -> list[IntervalBlockPair]:
    """Transforms the text blocks such that their number equals the number of depth intervals.

    If there are more depth intervals than text blocks, text blocks are splitted. When there
    are more text blocks than depth intervals, text blocks are merged. If the number of text blocks
    and depth intervals equals, we proceed with the pairing.

    Args:
        depth_intervals (List[Interval]): The depth intervals from the pdf.
        blocks (List[TextBlock]): Found textblocks from the pdf.

    Returns:
        List[IntervalBlockPair]: Pairing of text blocks and depth intervals.
    """
    if len(depth_intervals) <= 1:
        concatenated_block = TextBlock(
            [line for block in blocks for line in block.lines]
        )  # concatenate all text lines within a block; line separation flag does not matter here.
        depth_interval = depth_intervals[0] if len(depth_intervals) else None
        return [IntervalBlockPair(depth_interval=depth_interval, block=concatenated_block)]
    else:
        if len(blocks) < len(depth_intervals):
            blocks = _split_blocks_by_textline_length(blocks, target_split_count=len(depth_intervals) - len(blocks))

        if len(blocks) > len(depth_intervals):
            # create additional depth intervals with end & start value None to match the number of blocks
            depth_intervals.extend([AAboveBInterval(None, None) for _ in range(len(blocks) - len(depth_intervals))])

        return [
            IntervalBlockPair(depth_interval=depth_interval, block=block)
            for depth_interval, block in zip(depth_intervals, blocks, strict=False)
        ]


def _split_blocks_by_textline_length(blocks: list[TextBlock], target_split_count: int) -> list[TextBlock]:
    """Split textblocks without any geometric lines that separates them.

    The logic looks at the lengths of the text lines and cuts them off
    if there are textlines that are shorter than others.
    # TODO: Extend documentation about logic.

    Args:
        blocks (List[TextBlock]): Textblocks that are to be split.
        target_split_count (int): the number of splits that we'd like to happen (i.e. we'd like the total number of
                                  blocks to be increased by this number)

    Returns:
        List[TextBlock]: The split textblocks.
    """
    line_lengths = sorted([line.rect.x1 for block in blocks for line in block.lines[:-1]])
    if len(line_lengths) <= target_split_count:  # In that case each line is a block
        return [TextBlock([line]) for block in blocks for line in block.lines]
    else:
        cutoff_values = line_lengths[:target_split_count]  # all lines inside cutoff_values will be split line
        split_blocks = []
        current_block_lines = []
        for block in blocks:
            for line_index in range(block.line_count):
                line = block.lines[line_index]
                current_block_lines.append(line)
                if line_index < block.line_count - 1 and line.rect.x1 in cutoff_values:
                    split_blocks.append(TextBlock(current_block_lines))
                    cutoff_values.remove(line.rect.x1)
                    current_block_lines = []
            if current_block_lines:
                split_blocks.append(TextBlock(current_block_lines))
                current_block_lines = []
            if (
                block.is_terminated_by_line
            ):  # If block was terminated by a line, populate the flag to the last element of split_blocks.
                split_blocks[-1].is_terminated_by_line = True
        return split_blocks
