from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from langsmith import traceable

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


@traceable(name="PII Detection")
def redact_pii(text: str) -> tuple[str, bool]:
    """
    Detect and redact PII from user input.

    Returns:
        redacted_text
        pii_detected
    """

    results = analyzer.analyze(
        text=text,
        language="en",
    )

    if not results:
        return text, False

    redacted = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    )

    return redacted.text, True