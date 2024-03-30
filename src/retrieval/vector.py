from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever


def retrieve_top_k_documents(index: VectorStoreIndex, query: str, k: int = 5) -> List[str]:
    """
    Use a VectorIndexRetriever to retrieve the top `k` documents from a VectorStoreIndex.
    Returns a list of the top-k document texts, in decreasing order of relevance.
    """
    retriever = VectorIndexRetriever(index, similarity_top_k=k)
    return [node.node.get_content() for node in retriever.retrieve(query)]
