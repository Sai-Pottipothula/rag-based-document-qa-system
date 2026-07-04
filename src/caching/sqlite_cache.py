import hashlib
import time

from langsmith import traceable
from sqlitedict import SqliteDict

from src.models.answer import Answer
from src.models.cached_response import CachedResponse
from src.models.retrieved_chunk import RetrievedChunk


DB = SqliteDict(
    "data/cache/prompt_cache.sqlite",
    autocommit=True,
)


def cache_key(
    *,
    question: str,
    model: str,
    prompt_version: str,
    use_reranker: bool,
) -> str:
    raw = "|".join(
        [
            question.strip().lower(),
            model,
            prompt_version,
            str(use_reranker),
        ]
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


@traceable(name="SQLite Cache Lookup")
def cache_lookup(
    *,
    question: str,
    model: str,
    prompt_version: str,
    use_reranker: bool,
    ttl: int = 3600,
) -> CachedResponse | None:

    key = cache_key(
        question=question,
        model=model,
        prompt_version=prompt_version,
        use_reranker=use_reranker,
    )

    if key not in DB:
        return None

    item = DB[key]

    if time.time() - item.timestamp > ttl:
        del DB[key]
        return None

    return item


@traceable(name="SQLite Cache Store")
def cache_store(
    *,
    question: str,
    model: str,
    prompt_version: str,
    use_reranker: bool,
    answer: Answer,
    retrieved_chunks: list[RetrievedChunk],
) -> None:

    key = cache_key(
        question=question,
        model=model,
        prompt_version=prompt_version,
        use_reranker=use_reranker,
    )

    DB[key] = CachedResponse(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        timestamp=time.time(),
    )


@traceable(name="Clear SQLite Cache")
def clear_cache() -> None:
    DB.clear()


def cache_size() -> int:
    return len(DB)