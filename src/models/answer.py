from pydantic import BaseModel, Field

from src.models.citation import Citation
from src.models.retrieved_chunk import RetrievedChunk


class Answer(BaseModel):
    """
    Represents the final RAG answer.
    """

    answer: str

    citations: list[Citation]

    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        exclude=True,
    )