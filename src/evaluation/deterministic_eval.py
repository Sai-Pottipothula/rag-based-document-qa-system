from src.evaluation.utils import load_eval_set
from src.generation.rag_pipeline import generate_answer


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


def evaluate() -> None:
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

    print("\n========== DETERMINISTIC EVALUATION ==========\n")

    print(f"Questions                  : {total}")

    print(
        f"Answer Present             : "
        f"{answer_pass}/{total}"
    )

    print(
        f"Citation Present           : "
        f"{citation_pass}/{total}"
    )

    print(
        f"Valid Citation Format      : "
        f"{citation_format_pass}/{total}"
    )

    print(
        f"Minimum Answer Length      : "
        f"{length_pass}/{total}"
    )

    overall = (
        answer_pass
        + citation_pass
        + citation_format_pass
        + length_pass
    )

    possible = total * 4

    print(
        f"\nOverall Pass Rate          : "
        f"{overall / possible:.1%}"
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()