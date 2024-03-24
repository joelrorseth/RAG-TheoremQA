from pathlib import Path

OPENSTAX_URL = "https://openstax.org/books/prealgebra-2e/pages/1-introduction"

DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)

REFERENCE_DATA_PATH = DATA_PATH / "reference"
REFERENCE_DATA_PATH.mkdir(exist_ok=True)

INDEX_DATA_PATH = DATA_PATH / "index"
INDEX_DATA_PATH.mkdir(exist_ok=True)

OPENSTAX_DOWNLOAD_PATH = REFERENCE_DATA_PATH / "openstax"
OPENSTAX_DOWNLOAD_PATH.mkdir(exist_ok=True)

OPENSTAX_INDEX_PATH = INDEX_DATA_PATH / "openstax"
OPENSTAX_INDEX_PATH.mkdir(exist_ok=True)

HF_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
HF_EMBEDDING_MODEL_DIM = 384
