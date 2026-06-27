import os
import time

import cohere
from dotenv import load_dotenv

from src.models.rerank_response import RerankResponse
from src.models.reranked_chunk import RerankedChunk
from src.models.retrieval_response import RetrievalResponse
from src.retrieval.hybrid_search import hybrid_search

load_dotenv()


def create_reranker() -> cohere.Client:
    """
    Create the Cohere client.
    """

    return cohere.Client(
        os.getenv("COHERE_API_KEY"),
    )


def rerank(
    query: str,
    retrieve_k: int = 20,
    top_k: int = 6,
) -> RerankResponse:
    """
    Rerank retrieved chunks using Cohere.
    """

    start_time = time.perf_counter()

    retrieval_result: RetrievalResponse = hybrid_search(
        query=query,
        retrieve_k=retrieve_k,
        final_k=retrieve_k,
    )

    documents = [
        chunk.text
        for chunk in retrieval_result.chunks
    ]

    reranker = create_reranker()

    response = reranker.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_k,
    )

    reranked_chunks: list[RerankedChunk] = []

    for result in response.results:
        retrieved_chunk = retrieval_result.chunks[result.index]

        reranked_chunks.append(
            RerankedChunk(
                doc_id=retrieved_chunk.doc_id,
                chunk_id=retrieved_chunk.chunk_id,
                text=retrieved_chunk.text,
                score=retrieved_chunk.score,
                rerank_score=float(result.relevance_score),
            )
        )

    elapsed_time = time.perf_counter() - start_time

    return RerankResponse(
        query=query,
        chunks=reranked_chunks,
        reranker="rerank-v3.5",
        top_k=top_k,
        execution_time=elapsed_time,
    )


def main() -> None:
    query = input("Question: ")

    result = rerank(query)

    print("\nRERANK RESULTS\n")

    print(f"Execution Time : {result.execution_time:.4f} seconds")
    print(f"Top K          : {result.top_k}")
    print(f"Reranker       : {result.reranker}")
    print()

    for rank, chunk in enumerate(result.chunks, start=1):
        print("=" * 70)
        print(f"Rank            : {rank}")
        print(f"Retrieval Score : {chunk.score:.5f}")
        print(f"Rerank Score    : {chunk.rerank_score:.5f}")
        print(f"Document        : {chunk.doc_id}")
        print(f"Chunk ID        : {chunk.chunk_id}")
        print()
        print(chunk.text[:500])
        print()


if __name__ == "__main__":
    main()