import logging
import sys
import faiss
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from config import OPENSTAX_INDEX_PATH
from src.retrieval.parsing import load_openstax_subsections

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# TODO
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )


logging.info("Initializing faiss index")
d = 1536 # dimensions of text-ada-embedding-002
faiss_index = faiss.IndexFlatL2(d)

# Load documents
logging.info("Loading OpenStax into LlamaIndex documents")
openstax_subsection_chunks = load_openstax_subsections()
documents = [
    Document(
        text=chunk.content,
        metadata={
            "title": chunk.title,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "subsection_idx": chunk.index
        }
    ) for chunk in openstax_subsection_chunks
]


logging.info("Building FaissVectorStore")
vector_store = FaissVectorStore(faiss_index=faiss_index)

logging.info("Building StorageContext")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

logging.info("Building VectorStoreIndex")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# save index to disk
logging.info(f"Persisting index")
index.storage_context.persist(OPENSTAX_INDEX_PATH)

logging.info("Done building!")

# TODO: Load index from disk if available
# vector_store = FaissVectorStore.from_persist_dir("./storage")
# storage_context = StorageContext.from_defaults(
#     vector_store=vector_store, persist_dir="./storage"
# )
# index = load_index_from_storage(storage_context=storage_context)

# set Logging to DEBUG for more detailed outputs
# query_engine = index.as_query_engine()
# response = query_engine.query("formula to find the slope of a line when we have two points")

# set Logging to DEBUG for more detailed outputs
# query_engine = index.as_query_engine()
# response = query_engine.query(
#     "What did the author do after his time at Y Combinator?"
# )
