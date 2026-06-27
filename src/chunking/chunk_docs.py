from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.load_docs import load_documents
from src.models.chunk import Chunk
from src.models.document import Document
from src.utils.io import save_json


def create_splitter() -> RecursiveCharacterTextSplitter:
    """Create the text splitter."""

    return RecursiveCharacterTextSplitter(
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        chunk_size=800,
        chunk_overlap=100,
    )


def chunk_document(
    document: Document,
    splitter: RecursiveCharacterTextSplitter,
    start_chunk_id: int,
) -> tuple[list[Chunk], int]:
    """Split one document into chunks."""

    texts = splitter.split_text(document.text)

    chunks: list[Chunk] = []

    chunk_id = start_chunk_id

    for text in texts:
        chunks.append(
            Chunk(
                doc_id=document.doc_id,
                chunk_id=chunk_id,
                text=text,
            )
        )

        chunk_id += 1

    return chunks, chunk_id


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Chunk all documents."""

    splitter = create_splitter()

    all_chunks: list[Chunk] = []

    chunk_id = 0

    for document in documents:
        chunks, chunk_id = chunk_document(
            document=document,
            splitter=splitter,
            start_chunk_id=chunk_id,
        )

        all_chunks.extend(chunks)

    return all_chunks


def main() -> None:
    documents = load_documents(Path("data/raw"))

    chunks = chunk_documents(documents)

    save_json(
        chunks,
        Path("data/processed/chunks.json"),
    )

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")
    print("Saved chunks.json")

if __name__ == "__main__":
    main()