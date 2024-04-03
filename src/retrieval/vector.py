import json
import logging
from typing import Any, Dict, List
from collections import deque
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from config import REFERENCE_PDF_CONTENTS_FILENAME, REFERENCE_PDFS_PATH, RetrievalStrategy


def _basic_retrieve_top_k_documents(index: VectorStoreIndex, query: str, k: int = 5) -> List[str]:
    """
    Use a VectorIndexRetriever to retrieve the top `k` documents from a VectorStoreIndex.
    Returns a list of the top-k document texts, in decreasing order of relevance.
    """
    retriever = VectorIndexRetriever(index, similarity_top_k=k)
    return [node.node.get_content() for node in retriever.retrieve(query)]


def _build_document_from_nearby_lines(
    textbook_contents: List[List[List[str]]], metadata: Dict[str, Any], num_words_per_doc: int
) -> str:
    """
    Build a document string from textbook text that occurs adjacent to the line
    implicated in the original retrieved document (specified in `metadata`).
    This strategy starts with the retrieved line, then attempts to add lines
    occurring before or after it, alternating between the two until the new
    document string contains at least `num_words_per_doc` words.
    """
    # TODO: Currently assuming we don't have subsections in textbook_contents

    textbook_section_lines = textbook_contents[metadata["chapter_idx"]
                                               ][metadata["section_idx"]]

    # Add retrieved line, then try to add lines before and after
    retrieved_line: str = textbook_section_lines[metadata["line_idx"]]
    retrieved_lines = deque([retrieved_line])
    before_idx, after_idx = metadata["line_idx"]-1, metadata["line_idx"]+1
    retrieved_word_count = len(retrieved_line.split())

    try_before = True
    while (
        retrieved_word_count <= num_words_per_doc
        and (before_idx >= 0 or after_idx < len(textbook_section_lines))
    ):
        if try_before and before_idx >= 0:
            line_before: str = textbook_section_lines[before_idx]
            line_before_word_count = len(line_before.split())
            retrieved_word_count += line_before_word_count
            retrieved_lines.appendleft(line_before)
            before_idx -= 1

        if not try_before and after_idx < len(textbook_section_lines):
            line_after: str = textbook_section_lines[after_idx]
            line_after_word_count = len(line_after.split())
            retrieved_word_count += line_after_word_count
            retrieved_lines.append(line_after)
            after_idx += 1

        try_before = not try_before

    # Concatenate all retrieved lines into a document
    logging.info(
        f"Appending document with {retrieved_word_count} words"
    )
    return "\n".join(line for line in retrieved_lines)


def _build_document_from_section(
    textbook_contents: List[List[List[str]]], metadata: Dict[str, Any]
) -> str:
    """
    Build a document string from the text of the textbook section in which
    the line implicated in the original retrieved document (specified in
    `metadata`) occurs.
    """
    # TODO: Currently assuming we don't have subsections in textbook_contents
    return "\n".join(
        textbook_contents[metadata["chapter_idx"]][metadata["section_idx"]]
    )


def retrieve_top_k_documents(
    index: VectorStoreIndex, query: str, k: int,
    retrieval_strategy: RetrievalStrategy, num_words_per_doc: int
) -> List[str]:
    """
    Use a VectorIndexRetriever to retrieve the top `k` documents from a VectorStoreIndex.
    Returns a list of the top-k document texts, in decreasing order of relevance,
    and expanded to include a maximum of `num_words_per_doc` in each document
    (by including words that come before and after the hit).
    """
    retriever = VectorIndexRetriever(index, similarity_top_k=k)
    retrieved_documents = []

    for node in retriever.retrieve(query):
        textbook_content_file_path = REFERENCE_PDFS_PATH\
            / node.metadata["textbook_name"]\
            / REFERENCE_PDF_CONTENTS_FILENAME

        with open(textbook_content_file_path, 'r') as file:
            textbook_contents = json.load(file)
            new_document = ""

            if retrieval_strategy == RetrievalStrategy.Nearby:
                new_document = _build_document_from_nearby_lines(
                    textbook_contents, node.metadata, num_words_per_doc
                )
            elif retrieval_strategy == RetrievalStrategy.Section:
                new_document = _build_document_from_section(
                    textbook_contents, node.metadata
                )
            else:
                raise ValueError(
                    f"Unexpected retrieval strategy '{retrieval_strategy.value}'"
                )

            retrieved_documents.append(new_document)

    return retrieved_documents
