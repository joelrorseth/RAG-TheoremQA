from collections import defaultdict
from typing import Dict, List
from config import IndexingStrategy
from src.preprocessing.textbook import PreprocessedTextbook

IndexNameTextbookDict = Dict[str, List[PreprocessedTextbook]]


def _group_textbooks_by_subfield(
    textbooks: List[PreprocessedTextbook]
) -> Dict[str, List[PreprocessedTextbook]]:
    grouped_textbooks = defaultdict(list)
    for textbook in textbooks:
        grouped_textbooks[textbook.identifier.subfield].append(textbook)
    return grouped_textbooks


def _group_by_textbook_name(
    textbooks: List[PreprocessedTextbook]
) -> Dict[str, List[PreprocessedTextbook]]:
    grouped_textbooks = defaultdict(list)
    for textbook in textbooks:
        grouped_textbooks[textbook.identifier.name].append(textbook)
    return grouped_textbooks


def get_index_textbook_groupings(
    strategy: IndexingStrategy, textbooks: List[PreprocessedTextbook]
) -> IndexNameTextbookDict:
    if strategy == IndexingStrategy.Subfield:
        return _group_textbooks_by_subfield(textbooks)
    else:
        return _group_by_textbook_name(textbooks)
