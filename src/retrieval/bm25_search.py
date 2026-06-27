import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.models.chunk import Chunk
from src.models.retrieval_response import RetrievalResponse
from src.models.retrieved_chunk import RetrievedChunk
from src.utils.io import load_json


def create_bm25() -> tuple[BM25Okapi, list[Chunk]]:
    """
    Create the BM25 index.
    """

    chunks = load_json(
        Path("data/processed/chunks.json"),
        Chunk,
    )

    tokenized_chunks = [chunk.text.lower().split() for chunk in chunks]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25, chunks


def bm25_search(
    query: str,
    limit: int = 10,
) -> RetrievalResponse:
    """
    Retrieve relevant chunks using BM25.
    """

    start_time = time.perf_counter()

    bm25, chunks = create_bm25()

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:limit]

    retrieved_chunks: list[RetrievedChunk] = []

    for index in ranked_indices:
        chunk = chunks[index]

        retrieved_chunks.append(
            RetrievedChunk(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(scores[index]),
            )
        )

    elapsed_time = time.perf_counter() - start_time

    return RetrievalResponse(
        query=query,
        chunks=retrieved_chunks,
        retrieval_method="bm25",
        total_candidates=len(retrieved_chunks),
        execution_time=elapsed_time,
    )


def main() -> None:
    query = input("Question: ")

    result = bm25_search(query)

    print("\nBM25 SEARCH RESULTS\n")

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
