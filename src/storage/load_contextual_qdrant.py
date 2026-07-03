from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.models.embedded_chunk import EmbeddedChunk
from src.utils.io import load_json

COLLECTION_NAME = "rag_documents_contextual"
VECTOR_SIZE = 3072


def create_client() -> QdrantClient:
    """
    Create a Qdrant client.
    """

    return QdrantClient(
        host="localhost",
        port=6333,
    )


def recreate_collection(client: QdrantClient) -> None:
    """
    Delete and recreate the contextual collection.
    """

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )


def build_points(
    embedded_chunks: list[EmbeddedChunk],
) -> list[PointStruct]:
    """
    Convert embedded chunks into Qdrant points.
    """

    points: list[PointStruct] = []

    for chunk in embedded_chunks:
        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                },
            )
        )

    return points


def upload_points(
    client: QdrantClient,
    points: list[PointStruct],
) -> None:
    """
    Upload vectors into Qdrant.
    """

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


def main() -> None:
    client = create_client()

    recreate_collection(client)

    embedded_chunks = load_json(
        Path("data/processed/contextual_embedded_chunks.json"),
        EmbeddedChunk,
    )

    points = build_points(embedded_chunks)

    upload_points(
        client=client,
        points=points,
    )

    print(f"Uploaded {len(points)} contextual vectors to Qdrant")


if __name__ == "__main__":
    main()
