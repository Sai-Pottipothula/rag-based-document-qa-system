# RAG-Based Document Q&A System

A production-oriented Retrieval-Augmented Generation (RAG) system that enables users to ask natural language questions over a document collection and receive grounded answers with source citations.

The system combines hybrid retrieval, contextual embeddings, Cohere reranking, GPT-4.1, enterprise-grade evaluation, LangSmith observability, prompt caching, model fallback routing, prompt injection protection, and PII redaction to demonstrate production-ready AI application development.

---

## Highlights

- Hybrid Retrieval (Vector Search + BM25 + Reciprocal Rank Fusion)
- Contextual Embeddings
- Cohere Rerank for improved relevance
- GPT-4.1 answer generation with citations
- LangSmith tracing and observability
- Prompt cache using SQLite
- Automatic model fallback router
- Prompt A/B testing framework
- Enterprise evaluation suite
- Prompt injection guardrails
- Microsoft Presidio PII redaction
- FastAPI REST API
- Streamlit web interface
- Customer Security Brief

---

# Architecture

```text
                                ┌─────────────────────┐
                                │     Documents       │
                                │ PDF / HTML / Text   │
                                └─────────┬───────────┘
                                          │
                                          ▼
                            ┌─────────────────────────┐
                            │ Ingestion & Chunking    │
                            └─────────┬───────────────┘
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
        Standard Embeddings                   Contextual Embeddings
                  │                                       │
                  └───────────────────┬───────────────────┘
                                      ▼
                    ┌──────────────────────────────────────┐
                    │      Qdrant + PostgreSQL/pgvector    │
                    └───────────────────┬──────────────────┘
                                        ▼
                          Hybrid Retrieval (BM25 + Vector)
                                        │
                                        ▼
                          Reciprocal Rank Fusion (RRF)
                                        │
                                        ▼
                               Cohere Rerank
                                        │
                                        ▼
                              Prompt Guardrail
                                        │
                                        ▼
                               Query Extraction
                                        │
                                        ▼
                              Context Builder
                                        │
                                        ▼
                             PII Redaction (Presidio)
                                        │
                                        ▼
                         SQLite Prompt Cache Lookup
                                        │
                                        ▼
                     GPT-4.1 / GPT-4.1-mini Model Router
                                        │
                                        ▼
                           Answer with Source Citations
                                        │
                  ┌─────────────────────┴─────────────────────┐
                  ▼                                           ▼
             FastAPI API                              Streamlit UI
                                        │
                                        ▼
                              LangSmith Observability
```

---

# Features

## Retrieval

- Hybrid Search (Vector Search + BM25)
- Reciprocal Rank Fusion (RRF)
- Contextual Embeddings
- Cohere Rerank
- Source citations

---

## Generation

- GPT-4.1 answer generation
- Structured output
- Prompt versioning
- Grounded responses
- Citation-aware generation

---

## Performance

- SQLite prompt cache
- Automatic model fallback (GPT-4.1 → GPT-4.1-mini)
- Latency tracking
- Cache hit/miss tracking

---

## Observability

- LangSmith tracing
- Nested execution traces
- Retrieval metadata
- Reranker metadata
- Generation metadata
- Security metadata
- Cache/router metadata

---

## Evaluation

- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- Hit Rate
- Semantic Similarity
- LLM-as-a-Judge
- Deterministic evaluation
- Contextual Retrieval comparison
- Prompt A/B testing

---

## Security

- Prompt injection guardrails
- Enterprise attack classification
- Microsoft Presidio PII redaction
- Query extraction
- Security audit metadata
- Customer Security Brief

---

# Tech Stack

### AI

- OpenAI GPT-4.1
- OpenAI Embeddings
- Cohere Rerank

### Frameworks

- LangChain
- LangSmith

### Retrieval

- Qdrant
- PostgreSQL + pgvector
- BM25

### Backend

- Python
- FastAPI

### Frontend

- Streamlit

### Infrastructure

- Docker
- SQLite

---

# Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── docs/
│   └── Customer_Security_Brief.pdf
│
├── src/
│   ├── api/
│   ├── caching/
│   ├── chunking/
│   ├── embedding/
│   ├── evaluation/
│   ├── experiments/
│   ├── generation/
│   ├── ingestion/
│   ├── models/
│   ├── observability/
│   ├── retrieval/
│   ├── routing/
│   ├── security/
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

# Getting Started

## Clone

```bash
git clone https://github.com/<your-username>/rag-based-document-qa-system.git

cd rag-based-document-qa-system
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Create a `.env` file.

```text
OPENAI_API_KEY=
COHERE_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

---

## Start Infrastructure

```bash
docker compose up -d
```

---

## Ingest Documents

```bash
python -m src.ingestion.load_docs

python -m src.chunking.chunk_docs

python -m src.embedding.embed_chunks

python -m src.embedding.contextual_embed_chunks

python -m src.storage.load_qdrant

python -m src.storage.load_contextual_qdrant

python -m src.storage.load_pgvector
```

---

## Run the API

```bash
uvicorn src.api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Launch the UI

```bash
streamlit run src.ui.app.py
```

---

# Evaluation

Retrieval evaluation

```bash
python -m src.evaluation.retrieval_eval
```

Semantic evaluation

```bash
python -m src.evaluation.semantic_eval
```

LLM-as-a-Judge

```bash
python -m src.evaluation.llm_judge
```

Deterministic evaluation

```bash
python -m src.evaluation.deterministic_eval
```

Contextual retrieval comparison

```bash
python -m src.evaluation.compare_contextual
```

Prompt A/B testing

```bash
python -m src.experiments.prompt_ab_test

python -m src.experiments.report_generator
```

---

# Example Evaluation Results

| Metric | Standard | Contextual |
|---------|---------:|-----------:|
| Recall@1 | 0.725 | 0.775 |
| Recall@5 | 0.975 | 1.000 |
| Recall@10 | 0.975 | 1.000 |
| MRR | 0.824 | 0.869 |

---

# API Endpoints

| Method | Endpoint | Description |
|----------|----------------|------------------------------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/query` | Generate answer |
| POST | `/query/stream` | Streaming response |

---

# Documentation

The repository includes additional project documentation:

- **Customer Security Brief** (`docs/Customer_Security_Brief.pdf`)
  - Security architecture
  - Data flow
  - Prompt injection protection
  - PII redaction
  - Audit trail
  - Data retention
  - Vendor security checklist