"""Contains a dataclass for interval entries, which each define the start and end depth of a layer."""

from extraction.features.stratigraphy.interval.interval import AToBInterval
from extraction.features.stratigraphy.sidebarentry.sidebar_entry import SidebarEntry


class IntervalEntry(SidebarEntry[AToBInterval]):
    """Sidebar entry with the start depth and end depth of the layer defined on a single line, e.g. "1m - 2m"."""

    def __init__(self, interval: AToBInterval, page_number: int) -> None:
        super().__init__(interval, interval.rect, page_number)

    @property
    def start_value(self) -> float | None:
        return self.value.start.value if self.value.start is not None else None

    @property
    def end_value(self) -> float | None:
        return self.value.end.value if self.value.end is not None else None
