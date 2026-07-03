from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from src.chunking.chunk_docs import chunk_documents
from src.ingestion.load_docs import load_documents
from src.models.chunk import Chunk
from src.models.embedded_chunk import EmbeddedChunk
from src.utils.io import save_json

load_dotenv()


def create_embedding_model() -> OpenAIEmbeddings:
    """Create the embedding model."""

    return OpenAIEmbeddings(
        model="text-embedding-3-large",
    )


def embed_chunk(
    chunk: Chunk,
    embedding_model: OpenAIEmbeddings,
) -> EmbeddedChunk:
    """Embed one chunk."""

    vector = embedding_model.embed_query(chunk.text)

    return EmbeddedChunk(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        embedding=vector,
    )


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Embed all chunks."""

    embedding_model = create_embedding_model()

    embedded_chunks: list[EmbeddedChunk] = []

    for chunk in chunks:
        embedded_chunks.append(
            embed_chunk(
                chunk=chunk,
                embedding_model=embedding_model,
            )
        )

    return embedded_chunks


def main() -> None:
    documents = load_documents(Path("data/raw"))

    chunks = chunk_documents(documents)

    embedded_chunks = embed_chunks(chunks)

    save_json(
        embedded_chunks,
        Path("data/processed/embedded_chunks.json"),
    )

    print(f"Documents Loaded : {len(documents)}")
    print(f"Chunks Created   : {len(chunks)}")
    print(f"Embeddings Made  : {len(embedded_chunks)}")
    print("Saved embedded_chunks.json")


if __name__ == "__main__":
    main()
