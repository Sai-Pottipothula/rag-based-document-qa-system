from pydantic import BaseModel


class EvaluationExample(BaseModel):
    """Represents one evaluation example."""
    id: int
    question: str
    reference_answer: str
    expected_chunks: list[int]
