import json
from pathlib import Path

from src.retrieval.hybrid_search import hybrid_search


def load_eval_set() -> list[dict]:
    """
    Load the evaluation dataset.
    """

    with open(
        Path("data/evaluation/eval_set.json"),
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def recall_at_k(
    retrieved_docs: list[str],
    expected_doc: str,
    k: int,
) -> float:
    """
    Compute Recall@K.
    """

    return 1.0 if expected_doc in retrieved_docs[:k] else 0.0


def reciprocal_rank(
    retrieved_docs: list[str],
    expected_doc: str,
) -> float:
    """
    Compute Reciprocal Rank.
    """

    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc == expected_doc:
            return 1 / rank

    return 0.0


def evaluate(
    collection_name: str,
) -> dict:
    """
    Evaluate one retrieval collection.
    """

    examples = load_eval_set()

    recall1 = []
    recall5 = []
    recall10 = []
    mrr = []

    total_time = 0.0

    for example in examples:
        response = hybrid_search(
            query=example["question"],
            retrieve_k=20,
            final_k=10,
            collection_name=collection_name,
        )

        total_time += response.execution_time

        retrieved_docs = list(dict.fromkeys(chunk.doc_id for chunk in response.chunks))

        expected_doc = example["expected_document"]

        recall1.append(
            recall_at_k(
                retrieved_docs,
                expected_doc,
                1,
            )
        )

        recall5.append(
            recall_at_k(
                retrieved_docs,
                expected_doc,
                5,
            )
        )

        recall10.append(
            recall_at_k(
                retrieved_docs,
                expected_doc,
                10,
            )
        )

        mrr.append(
            reciprocal_rank(
                retrieved_docs,
                expected_doc,
            )
        )

    metrics = {
        "questions": len(examples),
        "recall1": sum(recall1) / len(recall1),
        "recall5": sum(recall5) / len(recall5),
        "recall10": sum(recall10) / len(recall10),
        "mrr": sum(mrr) / len(mrr),
        "avg_time": total_time / len(examples),
    }

    return metrics


def main() -> None:
    metrics = evaluate("rag_documents")

    print("\n========== RETRIEVAL EVALUATION ==========\n")

    print(f"Questions              : {metrics['questions']}")
    print(f"Recall@1               : {metrics['recall1']:.3f}")
    print(f"Recall@5               : {metrics['recall5']:.3f}")
    print(f"Recall@10              : {metrics['recall10']:.3f}")
    print(f"MRR                    : {metrics['mrr']:.3f}")
    print(f"Average Time (sec)     : {metrics['avg_time']:.3f}")


if __name__ == "__main__":
    main()
