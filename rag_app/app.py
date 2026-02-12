import streamlit as st
import pandas as pd
import os
import requests
from retrieval_pipeline import RetrievalPipeline
import config

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Nobel Prize RAG Assistant",
    page_icon="🏆",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🏆 Nobel Prize RAG")
    st.markdown("---")

    # Vector DB Status
    db_exists = os.path.exists(config.CHROMA_PERSIST_DIRECTORY)
    db_status = "🟢 Ready" if db_exists and os.listdir(config.CHROMA_PERSIST_DIRECTORY) else "🔴 Not Found"
    st.write(f"**Vector DB:** {db_status}")

    st.markdown("---")

    # Models
    st.subheader("🤖 Models")
    st.write(f"**LLM:** {config.OLLAMA_LLM_MODEL}")
    st.write(f"**Embeddings:** {config.OLLAMA_EMBEDDING_MODEL}")

    st.markdown("---")

    # Clear chat
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- MAIN HEADER ----------------
st.title("🏆 Nobel Prize RAG Assistant")
st.write("Ask questions about Nobel Prize winners, categories, and years.")

# ---------------- CHECK OLLAMA ----------------
def check_ollama():
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

if not check_ollama():
    st.error("🔴 Ollama is not running. Please run: `ollama serve`")
    st.stop()

# ---------------- CHECK VECTOR DB ----------------
if not os.path.exists(config.CHROMA_PERSIST_DIRECTORY) or not os.listdir(config.CHROMA_PERSIST_DIRECTORY):
    st.error("🔴 Vector database not found. Run: `python main.py ingest`")
    
    if os.path.exists(config.CSV_FILE_PATH):
        df = pd.read_csv(config.CSV_FILE_PATH)
        st.info(f" CSV found with {len(df)} rows.")
    else:
        st.warning("CSV file not found.")
    
    st.stop()

# ---------------- LOAD PIPELINE ----------------
@st.cache_resource
def load_pipeline():
    return RetrievalPipeline()

pipeline = load_pipeline()

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Display sources for assistant messages
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, doc in enumerate(msg["sources"], 1):
                    name = doc['metadata'].get('fullName', doc['metadata'].get('orgName', 'Unknown'))
                    st.markdown(f"**Source {i}:** {name}")
                    st.markdown(f"*Category: {doc['metadata'].get('category', 'N/A')}*")
                    st.markdown(f"*Year: {doc['metadata'].get('awardYear', 'N/A')}*")
                    st.markdown(f"```\n{doc['content'][:200]}...\n```")

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Ask about Nobel Prize..."):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                # Get answer + sources
                result = pipeline.ask_with_sources(prompt)
                st.markdown(result['answer'])

                # Save assistant message with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result.get('source_documents', [])
                })

                # Show sources in expander
                # if result.get('source_documents'):
                #     with st.expander(f"📚 Sources ({len(result['source_documents'])})"):
                #         for i, doc in enumerate(result['source_documents'], 1):
                #             name = doc['metadata'].get('fullName', doc['metadata'].get('orgName', 'Unknown'))
                #             st.markdown(f"**Source {i}:** {name}")
                #             st.markdown(f"*Category: {doc['metadata'].get('category', 'N/A')}*")
                #             st.markdown(f"*Year: {doc['metadata'].get('awardYear', 'N/A')}*")
                #             st.markdown(f"```\n{doc['content'][:200]}...\n```")

            except Exception as e:
                response = f" Error: {str(e)}"
                st.error(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "sources": []})

# ---------------- WELCOME SECTION ----------------
if not st.session_state.messages:
    st.markdown("---")
    st.markdown("### 🎯 Example Questions")
    st.markdown("""
    - Who won the Nobel Prize in Physics in 1921?
    - Tell me about Marie Curie
    - Nobel Prize winners in 2020
    - Which organizations won the Peace Prize?
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Ollama + LangChain + ChromaDB"
    "</div>",
    unsafe_allow_html=True
)
