from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """Represents a retrieved chunk."""
    doc_id: str
    chunk_id: int
    text: str
    score: float