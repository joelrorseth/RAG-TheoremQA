from typing import List
from pydantic import BaseModel
from config import TextbookIdentifier


class PreprocessedTextbookChunk(BaseModel):
    chapter_idx: int
    section_idx: int
    subsection_idx: int
    line_idx: int
    content: str


class PreprocessedTextbook(BaseModel):
    identifier: TextbookIdentifier
    chunks: List[PreprocessedTextbookChunk]
