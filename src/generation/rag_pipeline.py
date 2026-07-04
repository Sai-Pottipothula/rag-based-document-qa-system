from dotenv import load_dotenv
from langsmith import traceable

from src.caching.sqlite_cache import (
    cache_lookup,
    cache_store,
)
from src.models.answer import Answer
from src.models.retrieval_response import RetrievalResponse
from src.models.rerank_response import RerankResponse
from src.observability.tracing import (
    trace_cache_router,
    trace_generation,
    trace_security,
)
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.rerank import rerank
from src.routing.model_router import (
    PRIMARY_MODEL,
    route_llm,
)
from src.security.pii import redact_pii
from src.security.query_extractor import extract_query
from src.security.security_pipeline import secure_question

load_dotenv()

PROMPT_VERSION = "v1"


@traceable(name="Build Context")
def build_context(
    retrieval_result: RerankResponse | RetrievalResponse,
) -> str:
    """
    Build the retrieved context.
    """

    context = []

    for chunk in retrieval_result.chunks:

        context.append(
            f"""
Document: {chunk.doc_id}
Chunk ID: {chunk.chunk_id}

{chunk.text}
"""
        )

    return "\n\n".join(context)


@traceable(name="Build Prompt")
def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the RAG prompt.
    """

    return f"""
You are a Retrieval-Augmented Generation assistant.

Answer ONLY using the supplied context.

If the answer is not present in the context,
reply:

I don't know.

Return the answer together with citations.

Retrieved Context:

{context}

Question:

{question}
"""


@traceable(name="Hybrid RAG Pipeline")
def generate_answer(
    question: str,
    use_reranker: bool = True,
) -> Answer:
    """
    Generate a RAG answer.
    """

    # ==================================================
    # Security Guard
    # ==================================================

    guard_result = secure_question(question)

    if not guard_result.safe:

        trace_security(
            retrieval_query=question,
            pii_detected=False,
            attack_type=(
                guard_result.attack_type.value
                if guard_result.attack_type
                else None
            ),
            severity=guard_result.severity,
            matched_rule=guard_result.matched_rule,
        )

        return Answer(
            answer=(
                "Your request was blocked by the security guardrail.\n\n"
                f"Attack Type : {guard_result.attack_type.value if guard_result.attack_type else 'Unknown'}\n"
                f"Severity    : {guard_result.severity}\n"
                f"Matched Rule: {guard_result.matched_rule}\n"
                f"Reason      : {guard_result.reason}"
            ),
            citations=[],
            retrieved_chunks=[],
        )

    # ==================================================
    # Query Extraction
    # ==================================================

    retrieval_query = extract_query(question)

    # ==================================================
    # Retrieval
    # ==================================================

    retrieval_result = hybrid_search(
        retrieval_query,
    )

    if use_reranker:

        retrieval_result = rerank(
            query=retrieval_query,
            retrieval_result=retrieval_result,
        )

    # ==================================================
    # Context
    # ==================================================

    context = build_context(
        retrieval_result,
    )

    # ==================================================
    # Prompt Cache
    # ==================================================

    cached = cache_lookup(
        question=question,
        model=PRIMARY_MODEL,
        prompt_version=PROMPT_VERSION,
        use_reranker=use_reranker,
    )

    if cached is not None:

        answer = cached.answer
        answer.retrieved_chunks = cached.retrieved_chunks

        trace_cache_router(
            cache_hit=True,
            selected_model=PRIMARY_MODEL,
            fallback_used=False,
            generation_latency=0.0,
        )

        return answer

    # ==================================================
    # PII Redaction
    # ==================================================

    sanitized_question, pii_detected = redact_pii(
        question,
    )

    trace_security(
        retrieval_query=retrieval_query,
        pii_detected=pii_detected,
    )

    # ==================================================
    # Prompt
    # ==================================================

    prompt = build_prompt(
        question=sanitized_question,
        context=context,
    )

    trace_generation(
        model=PRIMARY_MODEL,
        prompt_version=PROMPT_VERSION,
        context_length=len(context),
        documents=[
            chunk.doc_id
            for chunk in retrieval_result.chunks
        ],
        chunk_ids=[
            chunk.chunk_id
            for chunk in retrieval_result.chunks
        ],
    )

    # ==================================================
    # LLM Router
    # ==================================================

    (
        answer,
        selected_model,
        fallback_used,
        latency,
    ) = route_llm(prompt)

    answer.retrieved_chunks = retrieval_result.chunks

    # ==================================================
    # Cache Store
    # ==================================================

    cache_store(
        question=question,
        model=PRIMARY_MODEL,
        prompt_version=PROMPT_VERSION,
        use_reranker=use_reranker,
        answer=answer,
        retrieved_chunks=retrieval_result.chunks,
    )

    trace_cache_router(
        cache_hit=False,
        selected_model=selected_model,
        fallback_used=fallback_used,
        generation_latency=latency,
    )

    return answer


def main() -> None:

    question = input("Question: ")

    answer = generate_answer(question)

    print("\nANSWER\n")
    print(answer.answer)

    print("\nCITATIONS\n")

    for citation in answer.citations:

        print(
            f"{citation.doc_id} | Chunk {citation.chunk_id}"
        )


if __name__ == "__main__":
    main()