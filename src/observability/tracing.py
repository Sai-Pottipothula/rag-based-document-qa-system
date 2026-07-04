from typing import Any

from langsmith import get_current_run_tree


def add_metadata(**metadata: Any) -> None:
    """
    Add metadata to the current LangSmith trace.
    """

    run_tree = get_current_run_tree()

    if run_tree is None:
        return

    if run_tree.metadata is None:
        run_tree.metadata = {}

    run_tree.metadata.update(metadata)


# ==========================================================
# QUERY PIPELINE
# ==========================================================

def trace_retrieval(
    *,
    query: str,
    collection: str,
    retrieval_method: str,
    retrieve_k: int,
    final_k: int,
    bm25_candidates: int,
    vector_candidates: int,
    rrf_candidates: int,
    returned_chunks: int,
    documents: list[str],
    chunk_ids: list[int],
    top_rrf_scores: list[float],
    latency: float,
) -> None:
    """
    Log retrieval metadata.
    """

    add_metadata(
        stage="retrieval",
        query=query,
        collection=collection,
        retrieval_method=retrieval_method,
        retrieve_k=retrieve_k,
        final_k=final_k,
        bm25_candidates=bm25_candidates,
        vector_candidates=vector_candidates,
        rrf_candidates=rrf_candidates,
        returned_chunks=returned_chunks,
        retrieved_documents=documents,
        retrieved_chunk_ids=chunk_ids,
        top_rrf_scores=top_rrf_scores,
        retrieval_latency=round(latency, 4),
    )


def trace_reranker(
    *,
    model: str,
    input_chunks: int,
    output_chunks: int,
    rerank_scores: list[float],
    latency: float,
) -> None:
    """
    Log reranker metadata.
    """

    add_metadata(
        stage="reranker",
        reranker_model=model,
        input_chunks=input_chunks,
        output_chunks=output_chunks,
        top_rerank_scores=rerank_scores,
        reranker_latency=round(latency, 4),
    )


def trace_generation(
    *,
    model: str,
    prompt_version: str,
    context_length: int,
    documents: list[str],
    chunk_ids: list[int],
) -> None:
    """
    Log generation metadata.
    """

    add_metadata(
        stage="generation",
        model=model,
        prompt_version=prompt_version,
        context_length=context_length,
        documents_used=documents,
        chunk_ids_used=chunk_ids,
    )


def trace_cache_router(
    *,
    cache_hit: bool,
    selected_model: str,
    fallback_used: bool,
    generation_latency: float,
) -> None:
    """
    Log prompt cache and model routing metadata.
    """

    add_metadata(
        stage="cache_router",
        cache_hit=cache_hit,
        selected_model=selected_model,
        fallback_used=fallback_used,
        generation_latency=round(generation_latency, 4),
    )


# ==========================================================
# EVALUATION PIPELINE
# ==========================================================

def trace_evaluation(
    *,
    # Retrieval
    recall1: float | None = None,
    recall5: float | None = None,
    recall10: float | None = None,
    mrr: float | None = None,
    hit_rate: float | None = None,

    # Semantic
    semantic_similarity: float | None = None,

    # LLM Judge
    helpfulness: float | None = None,
    faithfulness: float | None = None,
    harm: float | None = None,

    # Deterministic
    pass_rate: float | None = None,

    # Contextual Retrieval
    baseline_mrr: float | None = None,
    contextual_mrr: float | None = None,
) -> None:
    """
    Log evaluation metadata.
    """

    add_metadata(
        stage="evaluation",

        # Retrieval
        recall1=recall1,
        recall5=recall5,
        recall10=recall10,
        mrr=mrr,
        hit_rate=hit_rate,

        # Semantic
        semantic_similarity=semantic_similarity,

        # LLM Judge
        helpfulness=helpfulness,
        faithfulness=faithfulness,
        harm=harm,

        # Deterministic
        pass_rate=pass_rate,

        # Contextual Retrieval
        baseline_mrr=baseline_mrr,
        contextual_mrr=contextual_mrr,
    )
    
def trace_security(
    *,
    retrieval_query: str,
    pii_detected: bool,
    attack_type: str | None = None,
    severity: str | None = None,
    matched_rule: str | None = None,
) -> None:
    """
    Log security metadata.
    """

    add_metadata(
        stage="security",
        retrieval_query=retrieval_query,
        pii_detected=pii_detected,
        attack_type=attack_type,
        severity=severity,
        matched_rule=matched_rule,
    )