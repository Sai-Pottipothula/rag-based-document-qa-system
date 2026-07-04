from enum import Enum

from langsmith import traceable
from pydantic import BaseModel


class AttackType(str, Enum):
    PROMPT_INJECTION = "Prompt Injection"
    PROMPT_LEAK = "Prompt Leak"
    JAILBREAK = "Jailbreak"
    ROLE_MANIPULATION = "Role Manipulation"
    TOOL_ABUSE = "Tool Abuse"
    SECRET_EXTRACTION = "Secret Extraction"


class GuardrailResult(BaseModel):
    """
    Result of prompt guard evaluation.
    """

    safe: bool
    attack_type: AttackType | None = None
    severity: str | None = None
    matched_rule: str | None = None
    reason: str | None = None


RULES = [

    # ------------------------------------------------
    # Prompt Injection
    # ------------------------------------------------

    (
        AttackType.PROMPT_INJECTION,
        "high",
        [
            "ignore previous instructions",
            "ignore all previous instructions",
            "forget previous instructions",
            "disregard previous instructions",
        ],
    ),

    # ------------------------------------------------
    # Prompt Leak
    # ------------------------------------------------

    (
        AttackType.PROMPT_LEAK,
        "high",
        [
            "system prompt",
            "developer instructions",
            "hidden instructions",
            "show your prompt",
            "print your prompt",
            "reveal your prompt",
        ],
    ),

    # ------------------------------------------------
    # Jailbreak
    # ------------------------------------------------

    (
        AttackType.JAILBREAK,
        "critical",
        [
            "jailbreak",
            "dan mode",
            "do anything now",
            "developer mode",
        ],
    ),

    # ------------------------------------------------
    # Role Manipulation
    # ------------------------------------------------

    (
        AttackType.ROLE_MANIPULATION,
        "medium",
        [
            "pretend you are",
            "act as system",
            "act as developer",
        ],
    ),

    # ------------------------------------------------
    # Tool Abuse
    # ------------------------------------------------

    (
        AttackType.TOOL_ABUSE,
        "critical",
        [
            "run python",
            "run bash",
            "execute shell",
            "call every tool",
            "use every tool",
        ],
    ),

    # ------------------------------------------------
    # Secret Extraction
    # ------------------------------------------------

    (
        AttackType.SECRET_EXTRACTION,
        "critical",
        [
            "api key",
            "password",
            "private key",
            "secret",
            "token",
        ],
    ),
]


@traceable(name="Prompt Guard")
def check_prompt(
    question: str,
) -> GuardrailResult:
    """
    Detect prompt injection and jailbreak attempts.
    """

    text = question.lower()

    for attack_type, severity, patterns in RULES:

        for pattern in patterns:

            if pattern in text:

                return GuardrailResult(
                    safe=False,
                    attack_type=attack_type,
                    severity=severity,
                    matched_rule=pattern,
                    reason=f"{attack_type.value} detected",
                )

    return GuardrailResult(
        safe=True,
    )