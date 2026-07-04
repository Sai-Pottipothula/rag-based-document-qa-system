import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from src.evaluation.utils import load_eval_set
from src.experiments.prompts import PROMPT_V1, PROMPT_V2
from src.generation.rag_pipeline import build_context
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.rerank import rerank
from src.routing.model_router import route_llm

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


@traceable(name="Prompt A/B Test")
def evaluate_prompt(
    prompt_template: str,
    question: str,
    context: str,
    ground_truth: str,
) -> dict:

    prompt = prompt_template.format(
        question=question,
        context=context,
    )

    start = time.perf_counter()

    answer, model, fallback, _ = route_llm(prompt)

    latency = time.perf_counter() - start

    answer_embedding = embeddings.embed_query(
        answer.answer
    )

    gt_embedding = embeddings.embed_query(
        ground_truth
    )

    similarity = cosine_similarity(
        answer_embedding,
        gt_embedding,
    )

    return {
        "answer": answer.answer,
        "model": model,
        "fallback": fallback,
        "latency": latency,
        "similarity": similarity,
    }


def retrieve_context(question: str) -> str:
    """
    Retrieve once and reuse for both prompts.
    """

    retrieval = hybrid_search(question)

    retrieval = rerank(
        query=question,
        retrieval_result=retrieval,
    )

    return build_context(retrieval)


def main():

    examples = load_eval_set()

    results = []

    for example in examples:

        if example.expected_behavior != "answer":
            continue

        print("=" * 80)
        print(example.question)

        # Retrieval + rerank ONLY ONCE
        context = retrieve_context(
            example.question
        )

        v1 = evaluate_prompt(
            PROMPT_V1,
            example.question,
            context,
            example.ground_truth,
        )

        v2 = evaluate_prompt(
            PROMPT_V2,
            example.question,
            context,
            example.ground_truth,
        )

        results.append(
            {
                "question": example.question,
                "prompt_v1": v1,
                "prompt_v2": v2,
            }
        )

    output = Path(
        "data/evaluation/ab_tests/prompt_ab_results.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print()
    print("Saved to:")
    print(output)


if __name__ == "__main__":
    main()