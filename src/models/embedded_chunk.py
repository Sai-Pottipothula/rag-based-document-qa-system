from src.models.chunk import Chunk


class EmbeddedChunk(Chunk):
    """Represents an embedded chunk."""

    embedding: list[float]