from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from src.models.retrieval_response import RetrievalResponse
from src.models.retrieved_chunk import RetrievedChunk
from src.storage.load_qdrant import COLLECTION_NAME
import time

load_dotenv()


def create_embedding_model() -> OpenAIEmbeddings:
    """Create the embedding model."""

    return OpenAIEmbeddings(
        model="text-embedding-3-large",
    )


def create_client() -> QdrantClient:
    """Create a Qdrant client."""

    return QdrantClient(
        host="localhost",
        port=6333,
    )


def vector_search(
    query: str,
    limit: int = 10,
    collection_name: str = COLLECTION_NAME,
) -> RetrievalResponse:
    """
    Retrieve the most similar chunks using vector search.
    """

    start_time = time.perf_counter()

    embedding_model = create_embedding_model()

    query_vector = embedding_model.embed_query(query)

    client = create_client()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
    ).points

    retrieved_chunks: list[RetrievedChunk] = []

    for point in results:
        retrieved_chunks.append(
            RetrievedChunk(
                doc_id=point.payload["doc_id"],
                chunk_id=point.payload["chunk_id"],
                text=point.payload["text"],
                score=float(point.score),
            )
        )

    elapsed_time = time.perf_counter() - start_time

    return RetrievalResponse(
        query=query,
        chunks=retrieved_chunks,
        retrieval_method="vector",
        total_candidates=len(retrieved_chunks),
        execution_time=elapsed_time,
    )


def main() -> None:
    query = input("Question: ")

    result = vector_search(query)

    print("\nVECTOR SEARCH RESULTS\n")

    print(f"Execution Time : {result.execution_time:.4f} seconds")
    print(f"Candidates     : {result.total_candidates}")
    print()

    for rank, chunk in enumerate(result.chunks, start=1):
        print("=" * 70)
        print(f"Rank      : {rank}")
        print(f"Score     : {chunk.score:.4f}")
        print(f"Document  : {chunk.doc_id}")
        print(f"Chunk ID  : {chunk.chunk_id}")
        print()
        print(chunk.text[:500])
        print()


if __name__ == "__main__":
    main()
