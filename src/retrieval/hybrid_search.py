import time
from collections import defaultdict
from src.storage.load_qdrant import COLLECTION_NAME
from src.models.retrieval_response import RetrievalResponse
from src.models.retrieved_chunk import RetrievedChunk
from src.retrieval.bm25_search import bm25_search
from src.retrieval.vector_search import vector_search


def reciprocal_rank_fusion(
    bm25_chunks: list[RetrievedChunk],
    vector_chunks: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Fuse BM25 and Vector Search results using Reciprocal Rank Fusion.
    """

    rrf_scores: dict[int, float] = defaultdict(float)

    chunk_lookup: dict[int, RetrievedChunk] = {}

    # BM25 contribution
    for rank, chunk in enumerate(bm25_chunks):
        rrf_scores[chunk.chunk_id] += 1 / (k + rank + 1)
        chunk_lookup[chunk.chunk_id] = chunk

    # Vector contribution
    for rank, chunk in enumerate(vector_chunks):
        rrf_scores[chunk.chunk_id] += 1 / (k + rank + 1)

        if chunk.chunk_id not in chunk_lookup:
            chunk_lookup[chunk.chunk_id] = chunk

    fused_chunks: list[RetrievedChunk] = []

    for chunk_id, score in rrf_scores.items():
        chunk = chunk_lookup[chunk_id]

        fused_chunks.append(
            RetrievedChunk(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
            )
        )

    fused_chunks.sort(
        key=lambda chunk: chunk.score,
        reverse=True,
    )

    return fused_chunks


def hybrid_search(
    query: str,
    retrieve_k: int = 20,
    final_k: int = 10,
    collection_name: str = COLLECTION_NAME, 
) -> RetrievalResponse:
    """
    Perform Hybrid Retrieval using BM25 + Vector Search + RRF.
    """

    start_time = time.perf_counter()

    bm25_result = bm25_search(
        query=query,
        limit=retrieve_k,
    )

    vector_result = vector_search(
        query=query,
        limit=retrieve_k,
        collection_name=collection_name,
    )

    fused_chunks = reciprocal_rank_fusion(
        bm25_chunks=bm25_result.chunks,
        vector_chunks=vector_result.chunks,
    )

    elapsed_time = time.perf_counter() - start_time

    return RetrievalResponse(
        query=query,
        chunks=fused_chunks[:final_k],
        retrieval_method="hybrid_rrf",
        total_candidates=len(fused_chunks),
        execution_time=elapsed_time,
    )


def main() -> None:
    query = input("Question: ")

    result = hybrid_search(query)

    print("\nHYBRID SEARCH RESULTS\n")

    print(f"Execution Time : {result.execution_time:.4f} seconds")
    print(f"Candidates     : {result.total_candidates}")
    print()

    for rank, chunk in enumerate(result.chunks, start=1):
        print("=" * 70)
        print(f"Rank      : {rank}")
        print(f"RRF Score : {chunk.score:.5f}")
        print(f"Document  : {chunk.doc_id}")
        print(f"Chunk ID  : {chunk.chunk_id}")
        print()
        print(chunk.text[:500])
        print()


if __name__ == "__main__":
    main()