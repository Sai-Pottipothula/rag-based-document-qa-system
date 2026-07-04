from langsmith import traceable

from src.security.prompt_guard import (
    GuardrailResult,
    check_prompt,
)


@traceable(name="Security Pipeline")
def secure_question(
    question: str,
) -> GuardrailResult:
    """
    Apply prompt injection detection.

    Returns:
        GuardrailResult
    """

    return check_prompt(question)