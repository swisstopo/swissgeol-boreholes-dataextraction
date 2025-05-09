"""Module for finding LayerIdentifierSidebar instances in a borehole profile."""

import re

from extraction.features.utils.text.textline import TextLine

from ...base.sidebar_entry import LayerIdentifierEntry
from ..classes.layer_identifier_sidebar import LayerIdentifierSidebar
from ..utils.cluster import Cluster


class LayerIdentifierSidebarExtractor:
    """Class that finds LayerIdentifierSidebar instances in a borehole profile."""

    @classmethod
    def find_layer_identifier_sidebar_entries(cls, lines: list[TextLine]) -> list[LayerIdentifierEntry]:
        r"""Find the layer identifier sidebar entries.

        Regex explanation:
        - \b is a word boundary. This ensures that the match must start at the beginning of a word.
        - [\da-z]+ matches one or more (+) alphanumeric characters (\d for digits and a-z for lowercase letters).
        - \) matches a closing parenthesis. The backslash is necessary because parentheses are special characters
          in regular expressions, so we need to escape it to match a literal parenthesis.
        This regular expression will match strings like "1)", "2)", "a)", "b)", "1a4)", "6de)", etc.

        Args:
            lines (list[TextLine]): The lines to search for layer identifier entries.

        Returns:
            list[LayerIdentifierEntry]: The layer identifier sidebar entries.
        """
        entries = []
        for line in sorted(lines, key=lambda line: line.rect.y0):
            if len(line.words) > 0:
                # Only match in the first word of every line, to avoid e.g. matching with "cm)" in a material
                # description containing an expression like "(diameter max 6 cm)".
                first_word = line.words[0]
                regex = re.compile(r"\b[\da-z-]+\)")
                match = regex.match(first_word.text)
                if match and len(first_word.text) < 7:
                    entries.append(LayerIdentifierEntry(rect=first_word.rect, value=first_word.text))
        return entries

    @classmethod
    def from_lines(cls, lines: list[TextLine]) -> list[LayerIdentifierSidebar]:
        """Find layer identifier sidebars from text lines.

        TODO: Similar to AToBSidebarExtractor.find_in_words(). Refactoring may be desired.

        Args:
            lines (list[TextLine]): The text lines in the document

        Returns:
            list[LayerIdentifierSidebar]: The found layer identifier sidebar.
        """
        entries = cls.find_layer_identifier_sidebar_entries(lines)
        if not entries:
            return []

        clusters = Cluster[LayerIdentifierEntry].create_clusters(entries, lambda entry: entry.rect)

        sidebars = [LayerIdentifierSidebar(cluster.entries) for cluster in clusters if len(cluster.entries) >= 2]

        sidebars_by_length = sorted(
            [sidebar for sidebar in sidebars if sidebar],
            key=lambda sidebar: len(sidebar.entries),
            reverse=True,
        )

        result = []
        # Remove columns that are fully contained in a longer column
        # Also check that the columns are mostly ascending, both numerically and alphabetically (e.g. 2 < 3a < 3b
        # < 5 is valid). Some errors are tolerated, as the indicators (especially in Geneva) sometimes carry a
        # semantic meaning instead of being strictly ordered (e.g. geoquat/validation/9394.pdf), and we also need to
        # account for human or OCR mistakes. We only reject a column if the error ratio is bigger than the parameter
        # layer_identifier_acceptance_ratio specified in the config file.
        for sidebar in sidebars_by_length:
            if (
                not any(result_sidebar.rect().contains(sidebar.rect()) for result_sidebar in result)
                and sidebar.has_regular_progression()
            ):
                result.append(sidebar)

        return result
