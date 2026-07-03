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
    """
    Create the embedding model.
    """

    return OpenAIEmbeddings(
        model="text-embedding-3-large",
    )


def build_contextual_text(chunk: Chunk) -> str:
    """
    Build contextual text for embedding.

    Anthropic's contextual retrieval pattern:
    Include document metadata together with the chunk
    before creating embeddings.
    """

    return f"""
Document: {chunk.doc_id}

Content:

{chunk.text}
"""


def embed_chunk(
    chunk: Chunk,
    embedding_model: OpenAIEmbeddings,
) -> EmbeddedChunk:
    """
    Embed one chunk using contextual retrieval.
    """

    contextual_text = build_contextual_text(chunk)

    vector = embedding_model.embed_query(contextual_text)

    return EmbeddedChunk(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        embedding=vector,
    )


def embed_chunks(
    chunks: list[Chunk],
) -> list[EmbeddedChunk]:
    """
    Embed all chunks.
    """

    embedding_model = create_embedding_model()

    embedded_chunks: list[EmbeddedChunk] = []

    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        print(f"Embedding {index}/{total}")

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
        Path("data/processed/contextual_embedded_chunks.json"),
    )

    print()

    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {len(chunks)}")
    print(f"Embeddings: {len(embedded_chunks)}")

    print()

    print("Saved contextual_embedded_chunks.json")


if __name__ == "__main__":
    main()
