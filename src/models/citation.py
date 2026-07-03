from pydantic import BaseModel


class Citation(BaseModel):
    """Represents a citation returned by the RAG system."""

    doc_id: str
    chunk_id: int
