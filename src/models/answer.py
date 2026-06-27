from pydantic import BaseModel

from src.models.citation import Citation


class Answer(BaseModel):
    """Represents the final RAG answer."""

    answer: str
    citations: list[Citation]