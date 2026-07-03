from pathlib import Path

import trafilatura
from pypdf import PdfReader

from src.models.document import Document
from src.utils.io import save_json


def load_html(path: Path) -> Document | None:
    """Load a single HTML document."""

    html = path.read_text(encoding="utf-8")

    text = trafilatura.extract(html)

    if not text:
        print(f"Warning: Could not extract text from {path.name}")
        return None

    return Document(
        doc_id=path.name,
        text=text,
    )


def load_pdf(path: Path) -> Document | None:
    """Load a single PDF document."""

    try:
        reader = PdfReader(path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():
            print(f"Warning: Could not extract text from {path.name}")
            return None

        return Document(
            doc_id=path.name,
            text=text,
        )

    except Exception as e:
        print(f"Error loading {path.name}: {e}")
        return None


def load_documents(folder: Path) -> list[Document]:
    """Load all supported documents from a folder."""

    documents: list[Document] = []

    for file in folder.rglob("*.pdf"):
        document = load_pdf(file)

        if document:
            documents.append(document)

    for file in folder.rglob("*.html"):
        document = load_html(file)

        if document:
            documents.append(document)

    return documents


def main() -> None:
    documents = load_documents(Path("data/raw"))

    save_json(
        documents,
        Path("data/processed/documents.json"),
    )

    print(f"Loaded {len(documents)} documents")
    print("Saved documents.json")


if __name__ == "__main__":
    main()
