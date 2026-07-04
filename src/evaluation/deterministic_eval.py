import json
from pathlib import Path

from langsmith import traceable

from src.evaluation.utils import load_eval_set
from src.generation.rag_pipeline import generate_answer
from src.observability.tracing import trace_evaluation


def has_answer(answer: str) -> bool:
    """
    Check whether an answer is present.
    """

    return bool(answer.strip())


def has_citations(citations: list) -> bool:
    """
    Check whether at least one citation is returned.
    """

    return len(citations) > 0


def valid_citations(citations: list) -> bool:
    """
    Check whether every citation contains the required fields.
    """

    required_fields = {
        "doc_id",
        "chunk_id",
    }

    for citation in citations:

        if hasattr(citation, "model_dump"):
            citation = citation.model_dump()

        if not required_fields.issubset(citation.keys()):
            return False

    return True


def minimum_answer_length(
    answer: str,
    minimum_length: int = 20,
) -> bool:
    """
    Check that the answer is sufficiently long.
    """

    return len(answer.strip()) >= minimum_length


@traceable(name="Deterministic Evaluation")
def evaluate() -> dict:
    """
    Run deterministic evaluation.
    """

    examples = load_eval_set()

    answer_pass = 0
    citation_pass = 0
    citation_format_pass = 0
    length_pass = 0

    for example in examples:

        result = generate_answer(
            example.question,
            use_reranker=False,
        )

        if has_answer(result.answer):
            answer_pass += 1

        if has_citations(result.citations):
            citation_pass += 1

        if valid_citations(result.citations):
            citation_format_pass += 1

        if minimum_answer_length(result.answer):
            length_pass += 1

    total = len(examples)

    overall = (
        answer_pass
        + citation_pass
        + citation_format_pass
        + length_pass
    )

    possible = total * 4

    pass_rate = overall / possible

    metrics = {
        "questions": total,
        "answer_present": answer_pass,
        "citation_present": citation_pass,
        "valid_citation_format": citation_format_pass,
        "minimum_answer_length": length_pass,
        "overall_pass_rate": pass_rate,
    }

    trace_evaluation(
        pass_rate=pass_rate,
    )

    return metrics


def main() -> None:

    metrics = evaluate()

    output_path = Path(
        "data/evaluation/deterministic_metrics.json"
    )

    output_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print("\n========== DETERMINISTIC EVALUATION ==========\n")

    print(f"Questions                  : {metrics['questions']}")

    print(
        f"Answer Present             : "
        f"{metrics['answer_present']}/{metrics['questions']}"
    )

    print(
        f"Citation Present           : "
        f"{metrics['citation_present']}/{metrics['questions']}"
    )

    print(
        f"Valid Citation Format      : "
        f"{metrics['valid_citation_format']}/{metrics['questions']}"
    )

    print(
        f"Minimum Answer Length      : "
        f"{metrics['minimum_answer_length']}/{metrics['questions']}"
    )

    print(
        f"\nOverall Pass Rate          : "
        f"{metrics['overall_pass_rate']:.1%}"
    )

    print(f"\nMetrics saved to           : {output_path}")


if __name__ == "__main__":
    main()