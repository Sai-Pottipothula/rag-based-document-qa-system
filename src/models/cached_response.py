from src.models.answer import Answer
from src.models.retrieved_chunk import RetrievedChunk
from pydantic import BaseModel


class CachedResponse(BaseModel):
    """
    Cached RAG response.
    """

    answer: Answer
    retrieved_chunks: list[RetrievedChunk]
    timestamp: float