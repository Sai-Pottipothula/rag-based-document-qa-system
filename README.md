# RAG-Based Document Q&A System

An end-to-end Retrieval-Augmented Generation (RAG) system that enables users to ask questions over a collection of documents and receive accurate answers with source citations.

The project combines hybrid retrieval (Vector Search + BM25), contextual embeddings, Cohere reranking, and GPT-4.1 to generate grounded responses. It also includes an evaluation framework to measure retrieval performance and compare contextual retrieval against standard embeddings.

---

## Architecture

```text
                                ┌──────────────────────┐
                                │      Documents       │
                                │ PDF / HTML / Text    │
                                └──────────┬───────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │  Ingestion & Chunking   │
                              └──────────┬──────────────┘
                                         │
                         ┌───────────────┴────────────────┐
                         ▼                                ▼
              ┌───────────────────┐           ┌────────────────────┐
              │ Standard Embedding │           │ Contextual Embedding│
              └──────────┬────────┘           └──────────┬─────────┘
                         │                               │
                         └───────────────┬───────────────┘
                                         ▼
                          ┌──────────────────────────────┐
                          │      Vector Databases        │
                          │  Qdrant + PostgreSQL/pgvector│
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                              ┌─────────────────────────┐
                              │     Hybrid Retrieval    │
                              │    Vector Search + BM25 │
                              └──────────┬──────────────┘
                                         │
                                         ▼
                              ┌─────────────────────────┐
                              │    Cohere Reranker      │
                              └──────────┬──────────────┘
                                         │
                                         ▼
                              ┌─────────────────────────┐
                              │     GPT-4.1 Generator   │
                              └──────────┬──────────────┘
                                         │
                                         ▼
                              ┌─────────────────────────┐
                              │ Answer with Citations   │
                              └──────────┬──────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼                                             ▼
        ┌────────────────────┐                     ┌────────────────────┐
        │    FastAPI API     │                     │   Streamlit UI     │
        └────────────────────┘                     └────────────────────┘


                    ───────────── Evaluation Pipeline ─────────────

        Recall@1 • Recall@5 • Recall@10 • MRR • LLM-as-Judge
              Contextual Retrieval Comparison (Before vs After)
```

---

## Features

* Document ingestion and preprocessing
* Intelligent document chunking
* OpenAI embeddings
* Anthropic-style contextual embeddings
* Vector search with Qdrant
* BM25 lexical search
* Hybrid retrieval using Reciprocal Rank Fusion (RRF)
* Cohere reranking
* GPT-4.1 answer generation
* Source citations
* FastAPI REST API
* Streamlit chat interface
* Retrieval evaluation (Recall@1, Recall@5, Recall@10, MRR)
* LLM-as-a-Judge evaluation
* Contextual retrieval comparison

---

## Tech Stack

* Python
* LangChain
* OpenAI API
* Cohere API
* Qdrant
* PostgreSQL + pgvector
* FastAPI
* Streamlit
* Docker

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── src/
│   ├── api/
│   ├── chunking/
│   ├── embedding/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── models/
│   ├── retrieval/
│   ├── storage/
│   ├── ui/
│   └── utils/
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/<your-username>/rag-based-document-qa-system.git

cd rag-based-document-qa-system
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file.

```text
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
```

### Start the databases

```bash
docker compose up -d
```

### Process the documents

```bash
python -m src.ingestion.load_docs

python -m src.chunking.chunk_docs

python -m src.embedding.embed_chunks

python -m src.embedding.contextual_embed_chunks

python -m src.storage.load_qdrant

python -m src.storage.load_contextual_qdrant

python -m src.storage.load_pgvector
```

### Run the API

```bash
uvicorn src.api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

### Launch the UI

```bash
streamlit run src/ui/app.py
```

---

## Evaluation


Run the retrieval evaluation:
```bash
python -m src.evaluation.retrieval_eval
```
Run the LLM-as-a-Judge evaluation:
```bash
python -m src.evaluation.llm_judge
```
Compare standard and contextual retrieval:
```bash
python -m src.evaluation.compare_contextual
```

Example results:
| Metric    | Standard | Contextual |
| --------- | -------- | ---------- |
| Recall@1  | 0.725    | 0.775      |
| Recall@5  | 0.975    | 1.000      |
| Recall@10 | 0.975    | 1.000      |
| MRR       | 0.824    | 0.869      |

---

## API Endpoints

| Method | Endpoint        | Description                 |
| ------ | --------------- | --------------------------- |
| GET    | `/`             | API status                  |
| GET    | `/health`       | Health check                |
| POST   | `/query`        | Generate an answer          |
| POST   | `/query/stream` | Stream the generated answer |