import os
import re

import streamlit as st
from dotenv import load_dotenv

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama  # for local fallback

# -------------------------
# Setup
# -------------------------

load_dotenv()

st.set_page_config(
    page_title="Test Failure Triage Bot",
    page_icon="🛠️",
    layout="wide",
)

st.title("Test Failure Triage Bot with Chat History")
st.write("Paste failing test logs and chat with an AI SDET about root cause and next steps.")

# Groq API key (optional). If empty, use local Ollama instead.
api_key = st.text_input("Enter your GROQ API Key (leave blank to use local Ollama)", type="password")

# Choose LLM
if api_key:
    # WARNING: logs will be sent to Groq (external).
    llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
    st.info("Using Groq LLM (logs go to Groq servers).")
else:
    # Local LLM via Ollama (you must have Ollama + model installed)
    llm = ChatOllama(
    model="gemma3:1b",
    base_url="http://host.docker.internal:11434"
)

    st.info("Using local Ollama model (no logs sent to external LLM).")

# -------------------------
# Redaction: strip obvious secrets/PII
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
# Prompt + chain (LLM only, no RAG)
# -------------------------

triage_system_prompt = (
    "You are a senior QA automation and SDET experienced with CI/CD, Selenium, API tests, and backend logs.\n\n"
    "You will get:\n"
    "- Test name and environment\n"
    "- Logs (already partially redacted)\n"
    "- Chat history\n"
    "- Latest user question\n\n"
    "For EACH reply you must:\n"
    "1) Identify the most likely ROOT CAUSE category:\n"
    "   - product bug\n"
    "   - test automation bug\n"
    "   - environment/infra issue\n"
    "   - data issue\n"
    "   - flakiness/timing\n"
    "2) Quote the 2–5 most critical log lines (or patterns) that support your conclusion.\n"
    "3) Suggest clear NEXT STEPS for the QA/engineer.\n"
    "4) Respect follow-up questions that refer to previous answers.\n"
    "5) If logs are insufficient or ambiguous, say so clearly.\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", triage_system_prompt),
    MessagesPlaceholder("chat_history"),
    (
        "human",
        "Environment: {environment}\n\n"
        "Logs (redacted):\n"
        "{logs}\n\n"
        "User question:\n"
        "{input}"
    ),
])

output_parser = StrOutputParser()
base_chain = prompt | llm | output_parser

# -------------------------
# Chat history store (per session_id)
# -------------------------

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",       # which key is treated as the chat message
    history_messages_key="chat_history",  # must match MessagesPlaceholder
    # output_message_key is optional here since chain returns a string
)

# -------------------------
# UI: session, logs, metadata
# -------------------------

session_id = st.text_input("Session ID", value="default_session")

st.subheader("1️⃣ Test context and logs")

environment = st.text_input("Environment (e.g. QA, STG, PROD)", value=st.session_state.get("environment", ""))

raw_logs = st.text_area(
    "Raw logs / stacktrace (paste once, then ask multiple questions)",
    height=250,
    value=st.session_state.get("logs", ""),
    placeholder="Paste failing test logs, stacktrace, Jenkins console output, etc. here...",
)

if st.button("Set / Update logs for this session"):
    st.session_state["environment"] = environment
    st.session_state["logs"] = raw_logs
    st.success("Logs and context updated for this session.")

st.subheader("2️⃣ Ask questions about this failure")

session_history = get_session_history(session_id)

# Render existing chat history
for msg in session_history.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat-style input
user_input = st.chat_input("Ask a question about this failure")

if user_input:
    if not st.session_state.get("logs", "").strip():
        st.error("You must set logs first (paste logs and click 'Set / Update logs for this session').")
    else:
        # Show the user message in UI
        with st.chat_message("user"):
            st.markdown(user_input)

        safe_logs = redact_sensitive(st.session_state["logs"])

        # Call LLM with spinner so it doesn't feel stuck
        with st.chat_message("assistant"):
            with st.spinner("Analysing logs and previous conversation..."):
                response = conversation_chain.invoke(
                    {
                        "input": user_input,
                        "environment": st.session_state.get("environment", "") or "N/A",
                        "logs": safe_logs,
                    },
                    config={"configurable": {"session_id": session_id}},
                )
                st.markdown(response)
