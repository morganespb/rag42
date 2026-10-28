"""Pydantic models exchanged between the stages of the RAG pipeline."""

import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A piece of a file: its path and the character range it covers."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its correct sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources found for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources found for one question, plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and answers for a whole dataset."""

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """A piece of a file stored in the index, with its text."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
