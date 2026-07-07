"""Module for finding bounding boxes containing material descriptions."""

import dataclasses
import re

import pymupdf

from extraction.features.stratigraphy.sidebar.classes.a_above_b_sidebar import AAboveBSidebar
from extraction.features.stratigraphy.sidebar.classes.sidebar import Sidebar
from swissgeol_doc_processing.geometry.util import x_overlap_significant_smallest
from swissgeol_doc_processing.text.matching_params_analytics import MatchingParamsAnalytics
from swissgeol_doc_processing.text.textline import TextLine


@dataclasses.dataclass
class MaterialDescriptionExtractor:
    """Finds possible bounding boxes for material descriptions."""

    sidebar: Sidebar | None
    horizontal_text_lines: list[TextLine]
    language: str
    matching_params: dict
    analytics: MatchingParamsAnalytics = None

    def __post_init__(self):
        sidebar_zero_y0 = None
        if self.sidebar:
            if isinstance(self.sidebar, AAboveBSidebar) and len(self.sidebar.entries) > 1:
                first_entry = self.sidebar.entries[0]
                if first_entry.value <= 0.0:
                    sidebar_zero_y0 = first_entry.rect.y0
                else:
                    # extrapolate for AAboveBSidebars that don't have an explicit 0 depth
                    last_entry = self.sidebar.entries[-1]
                    if last_entry.value != first_entry.value:
                        slope = (last_entry.rect.y0 - first_entry.rect.y0) / (last_entry.value - first_entry.value)
                        sidebar_zero_y0 = first_entry.rect.y0 - slope * first_entry.value
                    else:
                        sidebar_zero_y0 = first_entry.rect.y0
            else:
                sidebar_zero_y0 = self.sidebar.rect.y0

        self.sidebar_zero_y0 = sidebar_zero_y0

    def find_candidates(self) -> list[pymupdf.Rect]:
        """Find all material description candidates on the page.

        Args:
            sidebar (Sidebar | None): The sidebar for which we want to find the material descriptions.

        Returns:
            list[pymupdf.Rect]: A list of candidate rectangles for material descriptions.
        """

        def check_y_condition(rect: pymupdf.Rect) -> bool:
            if self.sidebar and self.sidebar_zero_y0 is not None:
                return self.sidebar_zero_y0 < (rect.y0 + rect.y1) / 2 and rect.y0 < self.sidebar.rect.y1
            else:
                return True

        candidate_description = [line for line in self.horizontal_text_lines if check_y_condition(line.rect)]

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
            bounding_box = self._expand_lines_to_bounding_box(
                cluster=cluster,
                candidate_description=candidate_description,
                is_not_description=is_not_description,
                sorted_above=sorted_above,
            )
            if bounding_box is not None:
                candidate_rects.append(bounding_box)
        return candidate_rects

    def _expand_lines_to_bounding_box(
        self,
        cluster: list[TextLine],
        candidate_description: list[TextLine],
        is_not_description: list[TextLine],
        sorted_above: list[TextLine],
    ) -> pymupdf.Rect | None:
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
            return None

        # expand to include entire last block
        def can_extend_below(best_x0, best_y1, line: TextLine, x_tolerance: float = 5, line_gap: float = 10):
            if self.sidebar and self.sidebar.rect.contains(line.rect):
                # Don't extend using lines that entirely fall within the sidebar rect. Avoids false positives e.g. for
                # Geoquat A430.pdf. This workaround should no longer be necessary with a more robust clustering
                # approach.
                return False

            not_far_below_current_rect = line.rect.y0 < best_y1 + line_gap
            within_sidebar = self.sidebar and ((line.rect.y0 + line.rect.y1) / 2 < self.sidebar.rect.y1)

            if not_far_below_current_rect:  # noqa: SIM108
                # if immediately below the previous line, then allow for indentation or centered text
                max_x0 = (best_x0 + best_x1) / 2
            else:
                # else only consider text that is nicely left-aligned with the current box
                max_x0 = best_x0 + 2 * x_tolerance

            return (
                (best_x0 - x_tolerance < line.rect.x0 < max_x0)
                and (not_far_below_current_rect or within_sidebar)
                and (line.rect.y1 > best_y1)
                and len(line.text) > 4  # avoid OCR artefacts
            )

        def can_extend_above(best_x0, best_y0, line: TextLine, x_tolerance: float = 5, line_gap: float = 10):
            not_far_above_current_rect = line.rect.y1 > best_y0 - line_gap
            within_sidebar = (
                self.sidebar_zero_y0 is not None and (line.rect.y0 + line.rect.y1) / 2 > self.sidebar_zero_y0
            )

            return (
                (best_x0 - x_tolerance < line.rect.x0 < best_x0 + 2 * x_tolerance)
                and (not_far_above_current_rect or within_sidebar)
                and (line.rect.y0 < best_y0)
                and len(line.text) > 4  # avoid OCR artefacts
            )

        while line := next(
            (line for line in self.horizontal_text_lines if can_extend_below(best_x0, best_y1, line)), None
        ):
            best_x0 = min(best_x0, line.rect.x0)
            best_x1 = max(best_x1, line.rect.x1)
            best_y1 = line.rect.y1

        while next_line := next(
            (
                desc_line
                for desc_line in sorted_above
                if can_extend_above(best_x0, best_y0, desc_line)
                and not re.fullmatch(r"[\d\s.,\-/]+", desc_line.text.strip())
                and (
                    self.sidebar is not None
                    or not any(
                        other
                        for other in self.horizontal_text_lines
                        if other is not desc_line
                        and abs(other.rect.y0 - desc_line.rect.y0) < desc_line.rect.height
                        and (other.rect.x1 < best_x0 - 10 or other.rect.x0 > best_x1 + 10)
                    )
                )
            ),
            None,
        ):
            best_x0 = min(best_x0, next_line.rect.x0)
            best_x1 = max(best_x1, next_line.rect.x1)
            best_y0 = next_line.rect.y0

        return pymupdf.Rect(best_x0, best_y0, best_x1, best_y1)
