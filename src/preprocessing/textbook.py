from typing import List
from pydantic import BaseModel
from config import TextbookIdentifier


class PreprocessedTextbookSubsection(BaseModel):
    title: str
    content: str
    chapter: int
    section: int
    index: int


class PreprocessedTextbook(BaseModel):
    identifier: TextbookIdentifier
    subsections: List[PreprocessedTextbookSubsection]
