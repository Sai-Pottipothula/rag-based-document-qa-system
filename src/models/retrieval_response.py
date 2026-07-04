from pydantic import BaseModel

from src.models.retrieved_chunk import RetrievedChunk


class RetrievalResponse(BaseModel):
    query: str

    chunks: list[RetrievedChunk]

    retrieval_method: str

    total_candidates: int

    execution_time: float

    bm25_candidates: int | None = None

    vector_candidates: int | None = None