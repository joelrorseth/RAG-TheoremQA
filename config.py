import logging
from enum import Enum
from typing import Optional
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

EVAL_DATA_PATH = DATA_PATH / "evaluation"
EVAL_DATA_PATH.mkdir(exist_ok=True)

OUTPUTS_DATA_PATH = DATA_PATH / "outputs"
OUTPUTS_DATA_PATH.mkdir(exist_ok=True)

HF_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
HF_EMBEDDING_MODEL_DIM = 384

INDEX_CHUNK_SIZE = 256

HF_OCR_MODEL_NAME = "facebook/nougat-small"
HF_OCR_MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_OCR_TOKENS_PER_PAGE = 3500
MAX_OCR_PARALLEL_PAGES = 10

LLM_TEMPERATURE = 0.1

REFERENCE_PDF_TEXTBOOK_FILENAME = "textbook.pdf"
REFERENCE_PDF_OUTLINE_FILENAME = "outline.json"
REFERENCE_PDF_CONTENTS_FILENAME = "contents.json"
REFERENCE_PDF_PAGES_SUBDIR_NAME = "pages"
REFERENCE_PDF_PAGE_FILENAME_PREFIX = "page"
REFERENCE_PDF_PAGE_FILENAME_SUFFIX = ".mmd"


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
        name="stitz",
        source_url="https://stitz-zeager.com/szprecalculus07042013.pdf",
        subject="calculus"
    ),
    TextbookIdentifier(
        name="guichard",
        source_url="https://www.whitman.edu/mathematics/multivariable/multivariable.pdf",
        subject="calculus"
    ),
    # TextbookIdentifier(
    #     name="grinstead",
    #     source_url="https://www.whitman.edu/mathematics/multivariable/multivariable.pdf",
    #     subject="probability"
    # ),
    # TextbookIdentifier(
    #     name="huber",
    #     source_url="https://www.markhuberdatascience.org/_files/ugd/c2b9b6_8e0fbc80cfa64a0aa0c393840b0d50f8.pdf",
    #     subject="probability"
    # )
]


class IndexingStrategy(Enum):
    Subject = "subject"
    Textbook = "textbook"


class PromptingStrategy(Enum):
    Basic = "basic"
    COT = "cot"
    RAG_TOP2_500W = "rag_top2_500w"
    RAG_TOP2_200W = "rag_top2_200w"
    RAG_TOP1_500W = "rag_top1_500w"
    RAG_TOP1_200W = "rag_top1_200w"


class LLM(Enum):
    ChatGPT35 = "gpt-3.5-turbo"


class EvaluationDataset(Enum):
    TheoremQA = "theoremqa"


class IndexConfig(BaseModel):
    indexing_strategy: IndexingStrategy
    index_name: str


class Experiment(BaseModel):
    llm: LLM
    prompting_strategy: PromptingStrategy
    index_config: Optional[IndexConfig]
    evaluation_dataset: EvaluationDataset

    def to_string(self) -> str:
        s1 = self.llm.value
        s2 = self.prompting_strategy.value
        s3 = (
            f"{self.index_config.indexing_strategy.value}_{self.index_config.index_name}"
            if self.index_config is not None
            else "noindex"
        )
        s4 = self.evaluation_dataset.value
        return f"{s1}_{s2}_{s3}_{s4}"


class Prompt(BaseModel):
    user_prompt: str


class MultiRolePrompt(Prompt):
    system_prompt: str
