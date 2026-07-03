import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(
    page_title="Production RAG",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Production RAG Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching documents..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "question": question,
                },
            )

            response.raise_for_status()

            result = response.json()

        except requests.exceptions.RequestException as error:
            st.error(f"Unable to connect to the API.\n\n{error}")

            st.stop()

    answer = result["answer"]

    citations = result["citations"]

    assistant_message = f"{answer}\n\n### Citations\n"

    for citation in citations:
        assistant_message += f"- {citation['doc_id']} (Chunk {citation['chunk_id']})\n"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_message,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

        st.markdown("### 📚 Sources")

        for citation in citations:
            st.write(f"📄 **{citation['doc_id']}** | Chunk {citation['chunk_id']}")
