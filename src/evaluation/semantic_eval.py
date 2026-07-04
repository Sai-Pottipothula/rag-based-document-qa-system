import json
import math
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from src.evaluation.utils import load_eval_set
from src.generation.rag_pipeline import generate_answer
from src.observability.tracing import trace_evaluation

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    """
    Compute cosine similarity.
    """

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


@traceable(name="Semantic Evaluation")
def evaluate() -> dict:
    """
    Run semantic similarity evaluation.
    """

    examples = load_eval_set()

    similarities = []

    results = []

    for example in examples:

        # Only evaluate answerable questions.
        if example.expected_behavior != "answer":
            continue

        response = generate_answer(
            example.question,
            use_reranker=False,
        )

        answer_embedding = embeddings.embed_query(
            response.answer
        )

        reference_embedding = embeddings.embed_query(
            example.ground_truth
        )

        similarity = cosine_similarity(
            answer_embedding,
            reference_embedding,
        )

        similarities.append(similarity)

        results.append(
            {
                "question": example.question,
                "similarity": similarity,
            }
        )

        print("=" * 70)
        print(example.question)
        print(f"Similarity : {similarity:.3f}")
        print()

    similarities.sort()

    passed = sum(score >= 0.85 for score in similarities)

    metrics = {
        "questions": len(similarities),
        "average_similarity": sum(similarities) / len(similarities),
        "highest_similarity": max(similarities),
        "lowest_similarity": min(similarities),
        "median_similarity": similarities[len(similarities) // 2],
        "pass_rate": passed / len(similarities),
        "lowest_examples": sorted(
            results,
            key=lambda x: x["similarity"],
        )[:5],
    }

    trace_evaluation(
        semantic_similarity=metrics["average_similarity"],
    )

    return metrics


def main() -> None:

    metrics = evaluate()

    output_path = Path(
        "data/evaluation/semantic_metrics.json"
    )

    output_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print("\n========== SEMANTIC EVALUATION ==========\n")

    print(f"Questions              : {metrics['questions']}")
    print(f"Average Similarity     : {metrics['average_similarity']:.3f}")
    print(f"Highest Similarity     : {metrics['highest_similarity']:.3f}")
    print(f"Lowest Similarity      : {metrics['lowest_similarity']:.3f}")
    print(f"Median Similarity      : {metrics['median_similarity']:.3f}")
    print(f"Pass Rate (>=0.85)     : {metrics['pass_rate']:.3f}")

    print("\nLowest Similarity Questions:\n")

    for item in metrics["lowest_examples"]:
        print(
            f"{item['similarity']:.3f} | {item['question']}"
        )

    print(f"\nMetrics saved to       : {output_path}")


if __name__ == "__main__":
    main()