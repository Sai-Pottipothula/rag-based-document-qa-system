from pathlib import Path

from src.models.evaluation import EvaluationExample
from src.utils.io import load_json


def load_eval_set() -> list[EvaluationExample]:
    """
    Load the evaluation dataset.
    """

    return load_json(
        Path("data/evaluation/eval_set.json"),
        EvaluationExample,
    )