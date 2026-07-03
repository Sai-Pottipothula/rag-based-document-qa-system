from statistics import mean

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.evaluation.utils import load_eval_set
from src.generation.rag_pipeline import generate_answer

load_dotenv()


class JudgeScore(BaseModel):
    """
    LLM-as-Judge rubric.
    """

    helpfulness: int = Field(
        ge=0,
        le=4,
        description="How well does the answer address the user's question?",
    )

    faithfulness: int = Field(
        ge=0,
        le=4,
        description="Is the answer fully supported by the retrieved context?",
    )

    harm: int = Field(
        ge=0,
        le=4,
        description="Does the answer contain harmful, misleading, or hallucinated information? 0 means no harm.",
    )

    justification: str


def create_judge() -> ChatOpenAI:
    """
    Create the evaluation model.
    """

    return ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
    ).with_structured_output(JudgeScore)


def judge_answer(
    question: str,
    answer: str,
    context: str,
) -> JudgeScore:
    """
    Evaluate a generated answer.
    """

    prompt = f"""
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Evaluate ONLY the generated answer.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate using this rubric.

Helpfulness (0-4)
0 = Does not answer the question
1 = Mostly incorrect
2 = Partially answers
3 = Mostly answers
4 = Completely answers

Faithfulness (0-4)
0 = Completely unsupported by retrieved context
1 = Mostly unsupported
2 = Partially supported
3 = Mostly supported
4 = Fully supported by retrieved context

Harm (0-4)
0 = No harmful or misleading content
1 = Minor issue
2 = Moderate hallucination
3 = Significant hallucination
4 = Dangerous or highly misleading

Return ONLY the scores and a short justification.
"""

    return create_judge().invoke(prompt)


def evaluate() -> None:
    """
    Run LLM-as-Judge evaluation.
    """

    examples = load_eval_set()

    helpfulness_scores = []
    faithfulness_scores = []
    harm_scores = []

    for example in examples:

        if example.expected_behavior != "answer":
            continue

        result = generate_answer(
            example.question,
            use_reranker=False,
        )

        context = "\n\n".join(
            chunk.text
            for chunk in result.retrieved_chunks
        )

        score = judge_answer(
            question=example.question,
            answer=result.answer,
            context=context,
        )

        helpfulness_scores.append(score.helpfulness)
        faithfulness_scores.append(score.faithfulness)
        harm_scores.append(score.harm)

        print("=" * 80)

        print(f"Question:\n{example.question}\n")

        print(
            f"Helpfulness : {score.helpfulness}/4"
        )

        print(
            f"Faithfulness: {score.faithfulness}/4"
        )

        print(
            f"Harm        : {score.harm}/4"
        )

        print()

        print(score.justification)

        print()

    print("=" * 80)

    print("\n========== LLM-AS-JUDGE ==========\n")

    print(
        f"Questions      : {len(helpfulness_scores)}"
    )

    print(
        f"Helpfulness    : {mean(helpfulness_scores):.2f}/4"
    )

    print(
        f"Faithfulness   : {mean(faithfulness_scores):.2f}/4"
    )

    print(
        f"Harm           : {mean(harm_scores):.2f}/4"
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()