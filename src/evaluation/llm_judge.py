import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.generation.rag_pipeline import generate_answer

load_dotenv()


class JudgeScore(BaseModel):
    """LLM evaluation scores."""

    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)

    justification: str


def load_eval_set() -> list[dict]:
    """
    Load evaluation dataset.
    """

    with open(
        Path("data/evaluation/eval_set.json"),
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def create_llm() -> ChatOpenAI:
    """
    Create the judge model.
    """

    return ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
    ).with_structured_output(JudgeScore)


def judge_answer(
    question: str,
    answer: str,
) -> JudgeScore:
    """
    Ask GPT to judge the generated answer.
    """

    prompt = f"""
You are an expert evaluator.

Evaluate the following answer.

Question:
{question}

Answer:
{answer}

Score the answer from 1 to 5 for:

- Correctness
- Completeness
- Relevance

Also provide a short justification.
"""

    llm = create_llm()

    return llm.invoke(prompt)


def evaluate() -> None:
    """
    Run LLM-as-a-Judge evaluation.
    """

    examples = load_eval_set()

    correctness = []
    completeness = []
    relevance = []

    for example in examples:

        question = example["question"]

        answer = generate_answer(question,  use_reranker=False)

        score = judge_answer(
            question=question,
            answer=answer.answer,
        )

        correctness.append(score.correctness)
        completeness.append(score.completeness)
        relevance.append(score.relevance)

        print("=" * 70)
        print(question)
        print(answer.answer)
        print(score.justification)
        print()

    print("\n========== LLM JUDGE ==========\n")

    print(f"Questions        : {len(examples)}")
    print(f"Correctness      : {sum(correctness)/len(correctness):.2f}/5")
    print(f"Completeness     : {sum(completeness)/len(completeness):.2f}/5")
    print(f"Relevance        : {sum(relevance)/len(relevance):.2f}/5")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()