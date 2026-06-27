from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector

from src.models.embedded_chunk import EmbeddedChunk
from src.utils.io import load_json

DB_NAME = "rag_pgvector"
DB_USER = "pbsnarayana"
DB_HOST = "localhost"

TABLE_NAME = "documents"


def create_connection():
    """Create a PostgreSQL connection."""

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=DB_HOST,
    )

    register_vector(conn)

    return conn


def create_table(conn) -> None:
    """Create the documents table if it does not exist."""

    cur = conn.cursor()

    cur.execute(
        f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS {TABLE_NAME}
        (
            id INTEGER PRIMARY KEY,
            doc_id TEXT,
            text TEXT,
            embedding VECTOR(3072)
        );
        """
    )

    conn.commit()
    cur.close()


def insert_chunks(
    conn,
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    """Insert embedded chunks into PostgreSQL."""

    cur = conn.cursor()

    for chunk in embedded_chunks:
        cur.execute(
            f"""
            INSERT INTO {TABLE_NAME}
            (
                id,
                doc_id,
                text,
                embedding
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s
            )
            ON CONFLICT (id)
            DO NOTHING
            """,
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.text,
                chunk.embedding,
            ),
        )

    conn.commit()
    cur.close()


def main() -> None:
    conn = create_connection()

    create_table(conn)

    embedded_chunks = load_json(
        Path("data/processed/embedded_chunks.json"),
        EmbeddedChunk,
    )

    insert_chunks(
        conn=conn,
        embedded_chunks=embedded_chunks,
    )

    print(f"Inserted {len(embedded_chunks)} vectors into pgvector")

    conn.close()


if __name__ == "__main__":
    main()
