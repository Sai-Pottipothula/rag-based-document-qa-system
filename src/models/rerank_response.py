from pydantic import BaseModel

from src.models.reranked_chunk import RerankedChunk


class RerankResponse(BaseModel):
    query: str
    chunks: list[RerankedChunk]

    reranker: str
    top_k: int
    execution_time: float
