from pydantic import BaseModel


class Chunk(BaseModel):
    """Represents one chunk of a document."""

    doc_id: str
    chunk_id: int
    text: str
