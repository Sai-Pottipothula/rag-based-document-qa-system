from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.generation.rag_pipeline import generate_answer

app = FastAPI(
    title="Production RAG API",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Production RAG API Running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/query")
def query(request: QueryRequest):
    """
    Standard RAG endpoint.
    """

    answer = generate_answer(
        request.question,
    )

    return answer.model_dump()


@app.post("/query/stream")
def stream_query(request: QueryRequest):
    """
    Streaming endpoint.
    """

    answer = generate_answer(
        request.question,
    )

    def event_stream():
        for word in answer.answer.split():
            yield word + " "

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
    )