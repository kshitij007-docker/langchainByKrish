import os
import re

import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# Page config
# -------------------------

st.set_page_config(
    page_title="Test Failure Triage RAG Bot",
    page_icon="🛠️",
    layout="wide",
)

# -------------------------
# Env + embeddings
# -------------------------

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

st.title("🛠️ Test Failure Triage Bot (Logs + Chat History)")
st.write("Paste failing test logs, we index them into a vector store, then you chat over the logs (proper RAG).")

# -------------------------
# LLM selection: Groq (remote) or Ollama (local)
# -------------------------

api_key = st.text_input(
    "Enter your GROQ API Key (leave blank to use local Ollama)",
    type="password"
)

if api_key:
    llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
    st.info("Using Groq LLM (logs go to Groq servers). Use only safe/sanitised logs.")
else:
    llm = ChatOllama(model="gemma3:1b")
    st.info("Using local Ollama model (no logs sent to external LLM).")

# -------------------------
# Redaction: strip obvious secrets/PII before indexing
# -------------------------

SENSITIVE_PATTERNS = [
    r"Authorization:\s*Bearer\s+[^\s]+",
    r"Authorization:\s*[^\n]+",
    r"Set-Cookie:\s*[^\n]+",
    r"(?i)password\s*=\s*['\"][^'\"\n]+['\"]",
    r"(?i)pwd\s*=\s*['\"][^'\"\n]+['\"]",
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    r"[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+",
    r"\b[0-9a-fA-F]{32,64}\b",
]

def redact_sensitive(text: str) -> str:
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted

# -------------------------
# Session state init
# -------------------------

if "store" not in st.session_state:
    st.session_state.store = {}

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# -------------------------
# Chat history helper
# -------------------------

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# -------------------------
# RAG prompts
# -------------------------

contextualise_q_system_prompt = (
    "Given a chat history and the latest user question about the logs, "
    "which might reference context in that chat history, "
    "reformulate a standalone question that can be understood without the chat history. "
    "Do NOT answer the question, just rewrite it if needed, otherwise return as-is."
)

contextualise_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualise_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

triage_system_prompt = (
    "You are a senior QA automation and SDET experienced with CI/CD, Selenium, API tests, and backend logs.\n\n"
    "Use ONLY the provided context chunks (log segments) to answer. "
    "If the answer is not in the context, say you don't know.\n\n"
    "For EACH reply you must:\n"
    "1) Identify the most likely ROOT CAUSE category:\n"
    "   - product bug\n"
    "   - test automation bug\n"
    "   - environment/infra issue\n"
    "   - data issue\n"
    "   - flakiness/timing\n"
    "2) Quote the 2–5 most critical log lines (or patterns) from the context that support your conclusion.\n"
    "3) Suggest clear NEXT STEPS for the QA/engineer.\n"
    "4) Respect follow-up questions.\n"
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", triage_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# -------------------------
# Sidebar: session + controls
# -------------------------

session_id = st.text_input("Session ID", value="default_session")

st.sidebar.header("Controls")
if st.sidebar.button("Reset conversation + index"):
    st.session_state.store = {}
    st.session_state.retriever = None
    st.session_state.vectorstore = None
    st.session_state.indexed = False
    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

# -------------------------
# 1️⃣ Logs ingestion -> RAG index
# -------------------------

st.subheader("1️⃣ Provide context logs (RAG index)")

test_name = st.text_input(
    "Test name (optional)",
    value=st.session_state.get("test_name", "")
)
environment = st.text_input(
    "Environment (e.g. QA, STG, PROD)",
    value=st.session_state.get("environment", "")
)

raw_logs = st.text_area(
    "Raw logs / stacktrace (these will be split, embedded and stored in a vector DB)",
    height=250,
    value=st.session_state.get("raw_logs", ""),
    placeholder="Paste failing test logs, stacktrace, Jenkins console output, etc. here...",
)

if st.button("Index these logs for RAG"):
    if not raw_logs.strip():
        st.error("Please paste some logs before indexing.")
    else:
        with st.spinner("Redacting and indexing logs into Chroma..."):
            st.session_state["test_name"] = test_name
            st.session_state["environment"] = environment
            st.session_state["raw_logs"] = raw_logs

            safe_logs = redact_sensitive(raw_logs)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )

            # Create Document chunks with metadata
            splits = text_splitter.create_documents(
                texts=[safe_logs],
                metadatas=[{
                    "test_name": test_name or "N/A",
                    "environment": environment or "N/A",
                }]
            )

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )
            retriever = vectorstore.as_retriever()

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.indexed = True

        st.success("Logs indexed successfully. You can now chat with RAG over these logs.")

# -------------------------
# 2️⃣ Chat over indexed logs (RAG + history)
# -------------------------

st.subheader("2️⃣ Ask questions about this failure (RAG + chat history)")

if not st.session_state.indexed or st.session_state.retriever is None:
    st.info("Index some logs above first, then ask questions here.")
else:
    # Build RAG chain using stored retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        st.session_state.retriever,
        contextualise_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_message_key="answer",
    )

    session_history = get_session_history(session_id)

    # Render existing chat
    for msg in session_history.messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    user_input = st.chat_input("Ask a question about these logs")

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant log chunks and analysing..."):
                response = conversation_rag_chain.invoke(
                    {
                        "input": user_input,
                    },
                    config={"configurable": {"session_id": session_id}}
                )
                answer = response["answer"]
                st.markdown(answer)
