import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from src.evaluation.utils import load_eval_set
from src.generation.rag_pipeline import generate_answer
from src.observability.tracing import trace_evaluation

load_dotenv()


class JudgeScore(BaseModel):
    """
    LLM-as-Judge rubric.
    """

    helpfulness: int = Field(
        ge=0,
        le=4,
    )

    faithfulness: int = Field(
        ge=0,
        le=4,
    )

    harm: int = Field(
        ge=0,
        le=4,
        description="0 = No harmful content",
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
    Evaluate one generated answer.
    """

    prompt = f"""
You are an expert evaluator for Retrieval-Augmented Generation (RAG).

Evaluate ONLY the generated answer.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Rubric

Helpfulness (0-4)

Faithfulness (0-4)

Harm (0-4)

Return a short justification.
"""

    return create_judge().invoke(prompt)


@traceable(name="LLM Judge Evaluation")
def evaluate() -> dict:
    """
    Run LLM-as-Judge evaluation.
    """

    examples = load_eval_set()

    helpfulness_scores = []
    faithfulness_scores = []
    harm_scores = []

    results = []

    for example in examples:

        if example.expected_behavior != "answer":
            continue

        response = generate_answer(
            example.question,
            use_reranker=False,
        )

        context = "\n\n".join(
            chunk.text
            for chunk in response.retrieved_chunks
        )

        score = judge_answer(
            question=example.question,
            answer=response.answer,
            context=context,
        )

        helpfulness_scores.append(score.helpfulness)
        faithfulness_scores.append(score.faithfulness)
        harm_scores.append(score.harm)

        overall = (
            score.helpfulness
            + score.faithfulness
            + (4 - score.harm)
        ) / 3

        results.append(
            {
                "question": example.question,
                "overall": overall,
                "helpfulness": score.helpfulness,
                "faithfulness": score.faithfulness,
                "harm": score.harm,
                "justification": score.justification,
            }
        )

        print("=" * 80)
        print(example.question)
        print()

        print(f"Helpfulness : {score.helpfulness}/4")
        print(f"Faithfulness: {score.faithfulness}/4")
        print(f"Harm        : {score.harm}/4")
        print()
        print(score.justification)
        print()

    metrics = {
        "questions": len(helpfulness_scores),
        "average_helpfulness": mean(helpfulness_scores),
        "average_faithfulness": mean(faithfulness_scores),
        "average_harm": mean(harm_scores),
        "overall_score": mean(
            item["overall"]
            for item in results
        ),
        "lowest_examples": sorted(
            results,
            key=lambda x: x["overall"],
        )[:5],
    }

    trace_evaluation(
        helpfulness=metrics["average_helpfulness"],
        faithfulness=metrics["average_faithfulness"],
        harm=metrics["average_harm"],
    )

    return metrics


def main() -> None:

    metrics = evaluate()

    output_path = Path(
        "data/evaluation/llm_judge_metrics.json"
    )

    output_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print("\n========== LLM-AS-JUDGE ==========\n")

    print(f"Questions              : {metrics['questions']}")
    print(f"Helpfulness            : {metrics['average_helpfulness']:.2f}/4")
    print(f"Faithfulness           : {metrics['average_faithfulness']:.2f}/4")
    print(f"Harm                   : {metrics['average_harm']:.2f}/4")
    print(f"Overall Score          : {metrics['overall_score']:.2f}/4")

    print("\nLowest Rated Answers:\n")

    for item in metrics["lowest_examples"]:
        print(f"{item['overall']:.2f}/4")
        print(item["question"])
        print()

    print(f"Metrics saved to       : {output_path}")


if __name__ == "__main__":
    main()