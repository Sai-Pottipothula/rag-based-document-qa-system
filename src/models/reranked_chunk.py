from src.models.retrieved_chunk import RetrievedChunk


class RerankedChunk(RetrievedChunk):
    """Represents a reranked retrieved chunk."""

    rerank_score: float