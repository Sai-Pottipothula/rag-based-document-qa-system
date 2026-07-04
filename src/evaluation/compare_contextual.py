import json
from pathlib import Path

from langsmith import traceable

from src.evaluation.retrieval_eval import evaluate
from src.observability.tracing import trace_evaluation


NORMAL_COLLECTION = "rag_documents"
CONTEXTUAL_COLLECTION = "rag_documents_contextual"


def percentage_improvement(
    baseline: float,
    new: float,
) -> float:
    """
    Compute percentage improvement.
    """

    if baseline == 0:
        return 0.0

    return ((new - baseline) / baseline) * 100


@traceable(name="Contextual Retrieval Comparison")
def compare() -> dict:
    """
    Compare normal retrieval against contextual retrieval.
    """

    normal = evaluate(NORMAL_COLLECTION)

    contextual = evaluate(CONTEXTUAL_COLLECTION)

    comparison = {
        "normal": normal,
        "contextual": contextual,
        "improvements": {
            "recall1": percentage_improvement(
                normal["recall1"],
                contextual["recall1"],
            ),
            "recall5": percentage_improvement(
                normal["recall5"],
                contextual["recall5"],
            ),
            "recall10": percentage_improvement(
                normal["recall10"],
                contextual["recall10"],
            ),
            "mrr": percentage_improvement(
                normal["mrr"],
                contextual["mrr"],
            ),
            "hit_rate": percentage_improvement(
                normal["hit_rate"],
                contextual["hit_rate"],
            ),
        },
    }

    trace_evaluation(
        baseline_mrr=normal["mrr"],
        contextual_mrr=contextual["mrr"],
    )

    return comparison


def main() -> None:

    comparison = compare()

    output_path = Path(
        "data/evaluation/contextual_comparison.json"
    )

    output_path.write_text(
        json.dumps(comparison, indent=4),
        encoding="utf-8",
    )

    normal = comparison["normal"]
    contextual = comparison["contextual"]

    print("\n========== CONTEXTUAL RETRIEVAL COMPARISON ==========\n")

    print(
        f"{'Metric':<20}"
        f"{'Normal':>12}"
        f"{'Contextual':>15}"
    )

    print("-" * 50)

    print(
        f"{'Recall@1':<20}"
        f"{normal['recall1']:>12.3f}"
        f"{contextual['recall1']:>15.3f}"
    )

    print(
        f"{'Recall@5':<20}"
        f"{normal['recall5']:>12.3f}"
        f"{contextual['recall5']:>15.3f}"
    )

    print(
        f"{'Recall@10':<20}"
        f"{normal['recall10']:>12.3f}"
        f"{contextual['recall10']:>15.3f}"
    )

    print(
        f"{'MRR':<20}"
        f"{normal['mrr']:>12.3f}"
        f"{contextual['mrr']:>15.3f}"
    )

    print(
        f"{'Hit Rate':<20}"
        f"{normal['hit_rate']:>12.3f}"
        f"{contextual['hit_rate']:>15.3f}"
    )

    print(
        f"{'Avg Time (sec)':<20}"
        f"{normal['avg_time']:>12.3f}"
        f"{contextual['avg_time']:>15.3f}"
    )

    print("\n========== IMPROVEMENTS ==========\n")

    for metric, improvement in comparison["improvements"].items():
        print(f"{metric:<15}: {improvement:+.2f}%")

    if contextual["mrr"] > normal["mrr"]:
        print("\n🏆 Contextual Retrieval performed better.")
    elif contextual["mrr"] < normal["mrr"]:
        print("\n🏆 Standard Retrieval performed better.")
    else:
        print("\n🤝 Both retrieval methods performed equally.")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()