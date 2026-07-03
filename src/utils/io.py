import json
from pathlib import Path
from typing import Type

from pydantic import BaseModel


def save_json(data: list[BaseModel], path: Path) -> None:
    """
    Save a list of Pydantic models as JSON.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = []

    for item in data:
        obj = item.model_dump()

        # Remove invalid unicode characters
        for key, value in obj.items():
            if isinstance(value, str):
                obj[key] = value.encode("utf-8", errors="ignore").decode("utf-8")

        cleaned.append(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            cleaned,
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_json(path: Path, model: Type[BaseModel]):
    """
    Load JSON into Pydantic models.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [model.model_validate(item) for item in data]
