from src.evaluation.retrieval_eval import evaluate


NORMAL_COLLECTION = "rag_documents"
CONTEXTUAL_COLLECTION = "rag_documents_contextual"


def main() -> None:

    normal = evaluate(NORMAL_COLLECTION)

    contextual = evaluate(CONTEXTUAL_COLLECTION)

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
        f"{'Avg Time (sec)':<20}"
        f"{normal['avg_time']:>12.3f}"
        f"{contextual['avg_time']:>15.3f}"
    )


if __name__ == "__main__":
    main()