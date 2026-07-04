import re

from langsmith import traceable


@traceable(name="Query Extractor")
def extract_query(
    text: str,
) -> str:
    """
    Extract the most likely search query from user input.

    Strategy:
    - Prefer the last question in the input.
    - Remove empty lines.
    - Fall back to the full text.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Search from the bottom
    for line in reversed(lines):

        if "?" in line:
            return line

    # Fallback
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()