"""This module contains logic to validate AAboveBSidebar instances."""

import dataclasses

import rtree

from ..classes.a_above_b_sidebar import AAboveBSidebar
from ..classes.sidebar import SidebarNoise, noise_count


@dataclasses.dataclass
class AAboveBSidebarValidator:
    """Validation logic for instances of the AAboveBSidebar class.

    Args:
        noise_count_threshold (float): Noise count threshold deciding how much noise is allowed in a sidebar
                                       to be valid.
        noise_count_offset (int): Offset for the noise count threshold. Affects the noise count criterion.
                                  Effective specifically for sidebars with very few entries.
    """

    noise_count_threshold: float
    noise_count_offset: int

    def is_valid(self, sidebar_noise: SidebarNoise[AAboveBSidebar], corr_coef_threshold: float = 0.99) -> bool:
        """Checks whether the sidebar is valid.

        The sidebar is considered valid if:
        - The number of entries is at least 3.
        - Its noise_count is less than the noise count threshold
          time the number of entries minus the noise count offset.
        - The entries are strictly increasing.
        - The entries are linearly correlated with their vertical position.

        Args:
            sidebar_noise (SidebarNoise): The SidebarNoise wrapping the sidebar to validate.
            corr_coef_threshold (float): The minimal correlation coefficient for the column to be deemed valid.

        Returns:
            bool: True if the sidebar is valid, False otherwise.
        """
        # When too much other text is in the column, then it is probably not valid.
        # The quadratic behavior of the noise count check makes the check stricter for columns with few entries
        # than columns with more entries. The more entries we have, the less likely it is that we found them by chance.
        # TODO: Once evaluation data is of good enough qualities, we should optimize for the parameter below.

        sidebar = sidebar_noise.sidebar
        noise = sidebar_noise.noise_count
        if len(sidebar.entries) < 3:
            return False

        if noise > self.noise_count_threshold * (len(sidebar.entries) - self.noise_count_offset) ** 2:
            return False
        # Check if the entries are strictly increasing.
        if not sidebar.is_strictly_increasing():
            return False
        if sidebar.close_to_arithmetic_progression():
            return False

        corr_coef = sidebar.pearson_correlation_coef()

        return corr_coef and corr_coef > corr_coef_threshold

    def reduce_until_valid(
        self, sidebar_noise: SidebarNoise[AAboveBSidebar], line_rtree: rtree.index.Index
    ) -> SidebarNoise | None:
        """Removes entries from the depth column until it fulfills the is_valid condition.

        is_valid checks whether there is too much noise (i.e. other text) in the column and whether the entries are
        linearly correlated with their vertical position.

        Args:
            sidebar_noise (SidebarNoise): The SidebarNoise wrapping the AAboveBSidebar to validate.
            line_rtree (rtree.index.Index): Pre-built R-tree of all text lines on page for spatial queries.

        Returns:
            sidebar_noise | None : The current SidebarNoise with entries removed from Sidebar until it is valid
            and the recalculated noise_count or None.
        """
        while sidebar_noise.sidebar.entries:
            if self.is_valid(sidebar_noise):
                return sidebar_noise

            new_sidebar = sidebar_noise.sidebar.remove_entry_by_correlation_gradient()
            if not new_sidebar:
                return None

            new_noise_count = noise_count(new_sidebar, line_rtree)
            sidebar_noise = SidebarNoise(sidebar=new_sidebar, noise_count=new_noise_count)

        return None
