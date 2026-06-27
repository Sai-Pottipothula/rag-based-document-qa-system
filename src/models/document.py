from pydantic import BaseModel


class Document(BaseModel):
    """Represents a loaded source document."""
    doc_id: str
    text: str