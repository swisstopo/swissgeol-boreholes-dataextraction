"""Module for finding LayerIdentifierSidebar instances in a borehole profile."""

import re

from stratigraphy.lines.line import TextLine

from .layer_identifier_sidebar import LayerIdentifierEntry, LayerIdentifierSidebar


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
                    entries.append(LayerIdentifierEntry(first_word.rect, first_word.text))
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

        layer_identifier_sidebars = [LayerIdentifierSidebar([entries[0]])]
        for entry in entries[1:]:
            has_match = False
            for column in layer_identifier_sidebars:
                if column.can_be_appended(entry.rect):
                    column.entries.append(entry)
                    has_match = True
            if not has_match:
                layer_identifier_sidebars.append(LayerIdentifierSidebar([entry]))

            # only keep columns whose entries are not fully contained in a different column
            layer_identifier_sidebars = [
                column
                for column in layer_identifier_sidebars
                if all(not other.strictly_contains(column) for other in layer_identifier_sidebars)
            ]
            # check if the column rect is a subset of another column rect. If so, merge the entries and sort them by
            # y0.
            for column in layer_identifier_sidebars:
                for other in layer_identifier_sidebars:
                    if column != other and column.is_contained(other.rect()):
                        for entry in other.entries:
                            if entry not in column.entries:
                                column.entries.append(entry)
                        column.entries.sort(key=lambda entry: entry.rect.y0)
                        layer_identifier_sidebars.remove(other)
                        break
        layer_identifier_sidebars = [column for column in layer_identifier_sidebars if len(column.entries) > 2]
        return layer_identifier_sidebars
