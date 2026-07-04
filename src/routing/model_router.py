import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from langchain_openai import ChatOpenAI
from langsmith import traceable

from src.models.answer import Answer


PRIMARY_MODEL = "gpt-4.1"
FALLBACK_MODEL = "gpt-4.1-mini"


def create_llm(model_name: str) -> ChatOpenAI:
    """
    Create a ChatOpenAI model.
    """

    return ChatOpenAI(
        model=model_name,
        temperature=0,
    ).with_structured_output(Answer)


@traceable(name="Model Router")
def route_llm(
    prompt: str,
) -> tuple[Answer, str, bool, float]:
    """
    Route requests through the primary model.

    Falls back only for transient OpenAI API failures.

    Returns:
        answer
        selected_model
        fallback_used
        generation_latency
    """

    try:

        start = time.perf_counter()

        answer = create_llm(
            PRIMARY_MODEL,
        ).invoke(prompt)

        latency = time.perf_counter() - start

        return (
            answer,
            PRIMARY_MODEL,
            False,
            latency,
        )

    except (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    ):

        print(
            f"\nPrimary model ({PRIMARY_MODEL}) unavailable."
        )

        print(
            f"Falling back to {FALLBACK_MODEL}...\n"
        )

        start = time.perf_counter()

        answer = create_llm(
            FALLBACK_MODEL,
        ).invoke(prompt)

        latency = time.perf_counter() - start

        return (
            answer,
            FALLBACK_MODEL,
            True,
            latency,
        )