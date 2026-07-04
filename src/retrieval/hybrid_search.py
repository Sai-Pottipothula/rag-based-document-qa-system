import time
from collections import defaultdict
from langsmith import traceable
from src.observability.tracing import trace_retrieval
from src.storage.load_qdrant import COLLECTION_NAME
from src.models.retrieval_response import RetrievalResponse
from src.models.retrieved_chunk import RetrievedChunk
from src.retrieval.bm25_search import bm25_search
from src.retrieval.vector_search import vector_search
from src.observability.tracing import add_metadata
@traceable(name="Reciprocal Rank Fusion")
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
    add_metadata(
        stage="fusion",
        algorithm="RRF",
        rrf_k=k,
        bm25_input=len(bm25_chunks),
        vector_input=len(vector_chunks),
        fused_chunks=len(fused_chunks),
    )
    return fused_chunks


@traceable(name="Hybrid Retrieval")
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

    returned_chunks = fused_chunks[:final_k]

    trace_retrieval(
        query=query,
        collection=collection_name,
        retrieval_method="hybrid_rrf",
        retrieve_k=retrieve_k,
        final_k=final_k,
        bm25_candidates=len(bm25_result.chunks),
        vector_candidates=len(vector_result.chunks),
        rrf_candidates=len(fused_chunks),
        returned_chunks=len(returned_chunks),
        documents=[chunk.doc_id for chunk in returned_chunks],
        chunk_ids=[chunk.chunk_id for chunk in returned_chunks],
        top_rrf_scores=[
            round(chunk.score, 4)
            for chunk in returned_chunks
        ],
        latency=elapsed_time,
    )

    return RetrievalResponse(
        query=query,
        chunks=returned_chunks,
        retrieval_method="hybrid_rrf",
        total_candidates=len(fused_chunks),
        execution_time=elapsed_time,
        bm25_candidates=len(bm25_result.chunks),
        vector_candidates=len(vector_result.chunks),
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