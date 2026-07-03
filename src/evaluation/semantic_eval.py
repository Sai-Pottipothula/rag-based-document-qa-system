import math

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from src.evaluation.utils import load_eval_set
from src.generation.rag_pipeline import generate_answer

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


def evaluate() -> None:
    """
    Run semantic similarity evaluation.
    """

    examples = load_eval_set()

    similarities = []

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

        print("=" * 70)
        print(example.question)
        print(f"Similarity : {similarity:.3f}")
        print()

    print("\n========== SEMANTIC EVALUATION ==========\n")

    print(f"Questions             : {len(similarities)}")
    print(f"Average Similarity    : {sum(similarities)/len(similarities):.3f}")
    print(f"Highest Similarity    : {max(similarities):.3f}")
    print(f"Lowest Similarity     : {min(similarities):.3f}")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()