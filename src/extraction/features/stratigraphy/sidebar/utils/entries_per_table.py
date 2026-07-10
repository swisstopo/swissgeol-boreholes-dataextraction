"""Module for grouping sidebar entries by containment in detected table-like structures."""

import dataclasses
from typing import Generic, TypeVar

from extraction.features.stratigraphy.sidebarentry.sidebar_entry import SidebarEntry
from swissgeol_doc_processing.utils.table_detection import TableStructure

EntryT = TypeVar("EntryT", bound=SidebarEntry)


@dataclasses.dataclass
class TableEntries(Generic[EntryT]):
    """An optional table structure along with all possible entries within this structure."""

    table: TableStructure | None
    entries: list[EntryT]

    @classmethod
    def partition_entries(
        cls, table_structures: list[TableStructure], entries: list[EntryT]
    ) -> "list[TableEntries[EntryT]]":
        entries_per_table = {index: [] for index, table in enumerate(table_structures)}
        entries_no_table = []
        for entry in entries:
            table_found = False
            for index, table in enumerate(table_structures):
                if table.bounding_rect.intersects(entry.rect):
                    table_found = True
                    entries_per_table[index].append(entry)
            if not table_found:
                entries_no_table.append(entry)

        entry_partitions = [
            TableEntries(table_structures[table_index], entries) for table_index, entries in entries_per_table.items()
        ]
        entry_partitions.append(TableEntries(table=None, entries=entries_no_table))
        return entry_partitions
