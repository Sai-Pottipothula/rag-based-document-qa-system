from pydantic import BaseModel


class EvaluationExample(BaseModel):
    id: int
    category: str

    question: str

    expected_document: str | None = None

    expected_behavior: str

    ground_truth: str | None = None