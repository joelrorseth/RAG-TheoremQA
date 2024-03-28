import logging
import torch
from pathlib import Path
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)

REFERENCE_DATA_PATH = DATA_PATH / "reference"
REFERENCE_DATA_PATH.mkdir(exist_ok=True)

REFERENCE_HTML_DATA_PATH = REFERENCE_DATA_PATH / "html"
REFERENCE_HTML_DATA_PATH.mkdir(exist_ok=True)

REFERENCE_PDFS_PATH = REFERENCE_DATA_PATH / "pdf"
REFERENCE_PDFS_PATH.mkdir(exist_ok=True)

INDEX_DATA_PATH = DATA_PATH / "index"
INDEX_DATA_PATH.mkdir(exist_ok=True)

HF_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
HF_EMBEDDING_MODEL_DIM = 384

HF_OCR_MODEL_NAME = "facebook/nougat-small"
HF_OCR_MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_OCR_TOKENS_PER_PAGE = 3500
MAX_OCR_PARALLEL_PAGES = 10

REFERENCE_PDF_TEXTBOOK_FILENAME = "textbook.pdf"
REFERENCE_PDF_PAGES_SUBDIR_NAME = "pages"
REFERENCE_PDF_PAGE_FILENAME_PREFIX = "page"
REFERENCE_PDF_PAGE_FILENAME_SUFFIX = ".txt"


class TextbookIdentifier(BaseModel):
    name: str
    source_url: str
    subject: str


REFERENCE_HTML_TEXTBOOKS = [
    TextbookIdentifier(
        name="openstax",
        source_url="https://openstax.org/books/prealgebra-2e/pages/1-introduction",
        subject="algebra"
    )
]


REFERENCE_PDF_TEXTBOOKS = [
    TextbookIdentifier(
        name="precalculus",
        source_url="https://stitz-zeager.com/szprecalculus07042013.pdf",
        subject="calculus"
    )
]
