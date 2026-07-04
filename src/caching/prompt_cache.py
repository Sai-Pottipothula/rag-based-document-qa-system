import hashlib
import time
from langsmith import traceable
from src.models.answer import Answer
from src.models.cached_response import CachedResponse
from src.models.retrieved_chunk import RetrievedChunk


_CACHE: dict[str, CachedResponse] = {}


def cache_key(
    *,
    question: str,
    model: str,
    prompt_version: str,
    use_reranker: bool,
) -> str:
    """
    Generate a deterministic cache key.
    """

    raw_key = "|".join(
        [
            question.strip().lower(),
            model,
            prompt_version,
            str(use_reranker),
        ]
    )

    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


@traceable(name="Cache Lookup")
def cache_lookup(
    *,
    question: str,
    model: str,
    prompt_version: str,
    use_reranker: bool,
    ttl: int = 3600,
) -> CachedResponse | None:
    """
    Return a cached response if present and not expired.
    """

    key = cache_key(
        question=question,
        model=model,
        prompt_version=prompt_version,
        use_reranker=use_reranker,
    )

    item = _CACHE.get(key)

    if item is None:
        return None

    if time.time() - item.timestamp > ttl:
        del _CACHE[key]
        return None

    return item


@traceable(name="Cache Store")
def cache_store(
    *,
    question: str,
    model: str,
    prompt_version: str,
    use_reranker: bool,
    answer: Answer,
    retrieved_chunks: list[RetrievedChunk],
) -> None:
    """
    Store a RAG response in the cache.
    """

    key = cache_key(
        question=question,
        model=model,
        prompt_version=prompt_version,
        use_reranker=use_reranker,
    )

    _CACHE[key] = CachedResponse(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        timestamp=time.time(),
    )


@traceable(name="Clear Cache")
def clear_cache() -> None:
    """
    Clear all cached responses.
    """

    _CACHE.clear()


def cache_size() -> int:
    """
    Return the number of cached responses.
    """

    return len(_CACHE)