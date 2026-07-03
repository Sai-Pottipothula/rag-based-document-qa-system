from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.retrieval.hybrid_search import hybrid_search
from src.models.answer import Answer
from src.models.rerank_response import RerankResponse
from src.retrieval.rerank import rerank
from langsmith import traceable

load_dotenv()


def create_llm() -> ChatOpenAI:
    """
    Create the LLM.
    """

    return ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
    )

@traceable(name="Build Context")
def build_context(
    rerank_result: RerankResponse,
) -> str:
    """
    Build the retrieved context.
    """

    context = []

    for chunk in rerank_result.chunks:
        context.append(
            f"""
Document: {chunk.doc_id}
Chunk ID: {chunk.chunk_id}

{chunk.text}
"""
        )

    return "\n\n".join(context)

@traceable(name="Build Prompt")
def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the RAG prompt.
    """

    return f"""
You are a Retrieval-Augmented Generation assistant.

Answer ONLY using the supplied context.

If the answer is not present in the context,
reply:

I don't know.

Return the answer together with citations.

Retrieved Context:

{context}

Question:

{question}
"""

@traceable(name="Hybrid RAG Pipeline")
def generate_answer(
    question: str,
    use_reranker: bool = True,
) -> Answer:
    """
    Generate the final answer.
    """

    if use_reranker:
        retrieval_result = rerank(question)
    else:
        retrieval_result = hybrid_search(question)

    context = build_context(retrieval_result)

    prompt = build_prompt(
        question=question,
        context=context,
    )

    llm = create_llm().with_structured_output(Answer)

    answer = llm.invoke(prompt)

    answer.retrieved_chunks = retrieval_result.chunks

    return answer


def main() -> None:
    question = input("Question: ")

    answer = generate_answer(question)

    print("\nANSWER\n")

    print(answer.answer)

    print("\nCITATIONS\n")

    for citation in answer.citations:
        print(f"{citation.doc_id} | Chunk {citation.chunk_id}")


if __name__ == "__main__":
    main()

