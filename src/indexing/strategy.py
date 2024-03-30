from collections import defaultdict
from enum import Enum
from typing import Dict, List
from src.preprocessing.textbook import PreprocessedTextbook


class IndexingStrategy(Enum):
    Subject = "subject"
    Textbook = "textbook"


IndexNameTextbookDict = Dict[str, List[PreprocessedTextbook]]


def _group_textbooks_by_subject(
    textbooks: List[PreprocessedTextbook]
) -> Dict[str, List[PreprocessedTextbook]]:
    grouped_textbooks = defaultdict(list)
    for textbook in textbooks:
        grouped_textbooks[textbook.identifier.subject].append(textbook)
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
    if strategy == IndexingStrategy.Subject:
        return _group_textbooks_by_subject(textbooks)
    else:
        return _group_by_textbook_name(textbooks)
