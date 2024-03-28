import logging
import faiss
import shutil
from typing import List
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import HF_EMBEDDING_MODEL_DIM, HF_EMBEDDING_MODEL_NAME, INDEX_DATA_PATH
from src.indexing.strategy import IndexingStrategy, get_index_textbook_groupings
from src.preprocessing.textbook import PreprocessedTextbook


def build_index(index_name: str, strategy: IndexingStrategy,
                textbooks: List[PreprocessedTextbook], overwrite: bool = False):
    """
    Build a FAISS VectorStoreIndex for a set of textbooks.
    """
    index_path = INDEX_DATA_PATH / strategy.value / index_name

    # Handle existing index
    if index_path.exists() and index_path.is_dir():
        if not overwrite:
            logging.info(
                f"Already built '{strategy.value}/{index_name}' index"
            )
            return
        else:
            logging.info(f"Removing old '{strategy.value}/{index_name}' index")
            shutil.rmtree(index_path)
    else:
        index_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Building '{strategy.value}/{index_name}' index")

    logging.info("Configuring HuggingFaceEmbedding")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=HF_EMBEDDING_MODEL_NAME
    )

    logging.info("Initializing IndexFlatL2")
    faiss_index = faiss.IndexFlatL2(HF_EMBEDDING_MODEL_DIM)

    # TODO: Consider abstracting the 'subsection' chunk scope
    chunks = [ss for tb in textbooks for ss in tb.subsections]
    documents = [
        Document(
            text=chunk.content,
            metadata={
                "title": chunk.title,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "subsection_idx": chunk.index
            }
        ) for chunk in chunks
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


def build_all_indexes_for_strategy(
    strategy: IndexingStrategy, all_textbooks: List[PreprocessedTextbook],
    overwrite: bool = False
):
    logging.info(f"Building all indexes for '{strategy.value}' strategy")
    index_name_textbook_dict = get_index_textbook_groupings(
        strategy, all_textbooks
    )
    for index_name, index_textbooks in index_name_textbook_dict.items():
        build_index(index_name, strategy, index_textbooks, overwrite)


def build_all_indexes_for_all_strategies(
    all_textbooks: List[PreprocessedTextbook], overwrite: bool = False
):
    logging.info("Building all indexes for all strategies")
    for strategy in IndexingStrategy:
        build_all_indexes_for_strategy(strategy, all_textbooks, overwrite)
