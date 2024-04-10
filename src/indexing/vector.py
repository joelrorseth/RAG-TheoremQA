import logging
import faiss
import shutil
from typing import List
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import (
    EMBEDDING_MODELS,
    EmbeddingModel,
    IndexingStrategy,
    get_index_path
)
from src.indexing.strategy import get_index_textbook_groupings
from src.preprocessing.textbook import PreprocessedTextbook


def build_index(index_name: str, embedding_model: EmbeddingModel,
                strategy: IndexingStrategy, textbooks: List[PreprocessedTextbook],
                overwrite: bool = False):
    """
    Build a FAISS VectorStoreIndex for a set of textbooks.
    """
    index_path = get_index_path(index_name, embedding_model, strategy)

    # Handle existing index
    if index_path.exists() and index_path.is_dir():
        if not overwrite:
            logging.info(
                f"Already built '{strategy.value}/{index_name}' index, skipping"
            )
            return
        else:
            logging.info(f"Removing old '{strategy.value}/{index_name}' index")
            shutil.rmtree(index_path)
    else:
        index_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Building '{strategy.value}/{index_name}' index")

    logging.info(f"Configuring HuggingFaceEmbedding for {embedding_model.model}")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model.model
    )

    logging.info("Initializing IndexFlatL2")
    faiss_index = faiss.IndexFlatL2(embedding_model.dim)

    documents = [
        Document(
            text=chunk.content,
            metadata={
                "textbook_name": textbook.identifier.name,
                "chapter_idx": chunk.chapter_idx,
                "section_idx": chunk.section_idx,
                "subsection_idx": chunk.subsection_idx,
                "line_idx": chunk.line_idx
            }
        )
        for textbook in textbooks
        for chunk in textbook.chunks
    ]

    logging.info("Building FaissVectorStore")
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    logging.info("Building StorageContext")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logging.info("Building VectorStoreIndex")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    # Save index to disk
    logging.info(f"Persisting index to {index_path}")
    index.storage_context.persist(index_path)

    logging.info("Successfully created index!")


def build_all_vector_indexes_for_strategy(
    embedding_model: EmbeddingModel, strategy: IndexingStrategy,
    all_textbooks: List[PreprocessedTextbook], overwrite: bool = False
):
    logging.info(
        f"Building all {embedding_model.model} indexes for '{strategy.value}' strategy")
    index_name_textbook_dict = get_index_textbook_groupings(
        strategy, all_textbooks
    )
    for index_name, index_textbooks in index_name_textbook_dict.items():
        build_index(index_name, embedding_model,
                    strategy, index_textbooks, overwrite)


def build_all_vector_indexes(
    all_textbooks: List[PreprocessedTextbook], overwrite: bool = False
):
    logging.info("Building all indexes")
    for embedding_model in EMBEDDING_MODELS:
        for strategy in IndexingStrategy:
            build_all_vector_indexes_for_strategy(
                embedding_model, strategy, all_textbooks, overwrite
            )


def load_index(index_name: str, embedding_model: EmbeddingModel, strategy: IndexingStrategy) -> VectorStoreIndex:
    """
    Load a FAISS VectorStoreIndex.
    """
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model.model
    )
    index_path = get_index_path(index_name, embedding_model, strategy)
    vector_store = FaissVectorStore.from_persist_dir(index_path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=index_path
    )
    return load_index_from_storage(storage_context=storage_context)
