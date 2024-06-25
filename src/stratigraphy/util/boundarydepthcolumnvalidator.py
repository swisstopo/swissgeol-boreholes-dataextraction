"""This module contains logic to validate BoundaryDepthColumn instances."""

import dataclasses

from stratigraphy.util.depthcolumn import BoundaryDepthColumn
from stratigraphy.util.depthcolumnentry import DepthColumnEntry
from stratigraphy.util.line import TextWord


@dataclasses.dataclass
class BoundaryDepthColumnValidator:
    """Validation logic for instances of the BoundaryDepthColumn class.

    Args:
        all_words (list[TextLine]): A list of all text lines on the page.
        noise_count_threshold (float): Noise count threshold deciding how much noise is allowed in a column
                                       to be valid.
        noise_count_offset (int): Offset for the noise count threshold. Affects the noise count criterion.
                                  Effective specifically for depth columns with very few entries.
    """

    all_words: list[TextWord]
    noise_count_threshold: float
    noise_count_offset: int

    def is_valid(self, column: BoundaryDepthColumn, corr_coef_threshold: float = 0.99) -> bool:
        """Checks whether the depth column is valid.

        The depth column is considered valid if:
        - The number of entries is at least 3.
        - The number of words that intersect with the depth column entries is less than the noise count threshold
          time the number of entries minus the noise count offset.
        - The entries are strictly increasing.
        - The entries are linearly correlated with their vertical position.

        Note: The noise count criteria may require a rehaul. Some depth columns are not recognized as valid
        even though they are.

        Args:
            column (BoundaryDepthColumn): The depth column to validate.
            corr_coef_threshold (float): The minimal correlation coefficient for the column to be deemed valid.

        Returns:
            bool: True if the depth column is valid, False otherwise.
        """
        if len(column.entries) < 3:
            return False

        # When too much other text is in the column, then it is probably not valid.
        # The quadratic behavior of the noise count check makes the check stricter for columns with few entries
        # than columns with more entries. The more entries we have, the less likely it is that we found them by chance.
        # TODO: Once evaluation data is of good enough qualities, we should optimize for the parameter below.
        if (
            column.noise_count(self.all_words)
            > self.noise_count_threshold * (len(column.entries) - self.noise_count_offset) ** 2
        ):
            return False
        # Check if the entries are strictly increasing.
        if not column.is_strictly_increasing():
            return False

        corr_coef = column.pearson_correlation_coef()

        return corr_coef and corr_coef > corr_coef_threshold

    def reduce_until_valid(self, column: BoundaryDepthColumn) -> BoundaryDepthColumn:
        """Removes entries from the depth column until it fulfills the is_valid condition.

        is_valid checks whether there is too much noise (i.e. other text) in the column and whether the entries are
        linearly correlated with their vertical position.

        Args:
            column (BoundaryDepthColumn): The depth column to validate

        Returns:
            BoundaryDepthColumn: The current depth column with entries removed until it is valid.
        """
        while column:
            if self.is_valid(column):
                return column
            elif self.correct_OCR_mistakes(column) is not None:
                return self.correct_OCR_mistakes(column)
            else:
                column = column.remove_entry_by_correlation_gradient()

    def correct_OCR_mistakes(self, column: BoundaryDepthColumn) -> BoundaryDepthColumn | None:
        """Corrects OCR mistakes in the depth column entries.

        Loops through all values and corrects common OCR mistakes for the given entry. Then, the column with the
        highest pearson correlation coefficient is selected and checked for validity.

        This is useful if one or more entries have an OCR mistake, and the column is not valid because of it.

        Currently, there is no limit on the number of corrections per depth column. Indeed, there are examples of depth
        columns with multiple OCR errors on different depth values. On the other hand, allowing an unlimited number of
        corrections increases the risk, that a random column of different values is incorrectly accepted as a depth
        column after making the corrections, especially if the column has a low number of entries. A more robust
        solution might be to allow corrections on less than 50% of all entries, or something similar. However, we
        currently don't have enough examples to properly tune this parameter.

        Note: Common mistakes should be extended as needed.

        Args:
            column (BoundaryDepthColumn): The depth column to validate

        Returns:
            BoundaryDepthColumn | None: The corrected depth column, or None if no correction was possible.
        """
        new_columns = [BoundaryDepthColumn()]
        for entry in column.entries:
            new_columns = [
                BoundaryDepthColumn([*column.entries, DepthColumnEntry(entry.rect, new_value)])
                for column in new_columns
                for new_value in _value_alternatives(entry.value)
            ]
            # Immediately require strictly increasing values, to avoid exponential complexity when many implausible
            # alternative values are suggested
            new_columns = [column for column in new_columns if column.is_strictly_increasing()]

        if len(new_columns):
            best_column = max(new_columns, key=lambda column: column.pearson_correlation_coef())

            # We require a higher correlation coefficient when we've already corrected a mistake.
            if self.is_valid(best_column, corr_coef_threshold=0.999):
                return best_column

        return None


def _value_alternatives(value: float) -> set[float]:
    """Corrects frequent OCR errors in depth column entries.

    Args:
        value (float): The depth values to find plausible alternatives for

    Returns:
        set(float): all plausible values (including the original one)
    """
    alternatives = {value}
    # In older documents, OCR sometimes mistakes 1 for 4
    alternatives.add(float(str(value).replace("4", "1")))

    # replace a pattern such as '.80' with '0.80'. These cases are already converted
    # to '80.0' when depth entries are recognized.
    if value.is_integer():
        alternatives.add(value / 100)

    return alternatives
