import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq

# -------------------------
# 1. Configuration & Setup
# -------------------------
load_dotenv()

# Fix for HuggingFace Token
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

st.set_page_config(page_title="QA Ops Assistant", layout="wide")

# -------------------------
# 2. Embeddings Model
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# 3. Streamlit UI Headers
# -------------------------
st.title("🚀 QA Ops Assistant")
st.caption("AI-Powered Test Generation, Log Analysis & Infrastructure Lookup")

# Sidebar for Setup (Clean UI)
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Enter GROQ API Key", type="password")
    
    # Session ID Handling
    session_id = st.text_input("Session ID", value="default_session")
    
    # Manage Session State for History
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf","txt","json"], accept_multiple_files=True)


if not api_key:
    st.warning("Please enter your GROQ API Key in the sidebar to continue.")
    st.stop()

# -------------------------
# 4. Initialize LLM
# -------------------------
llm = ChatGroq(
    api_key=api_key, 
    model_name="llama-3.1-8b-instant",
    temperature=0.1 
)

# -------------------------
# 5. History Management
# -------------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# -------------------------
# 6. PDF Processing
# -------------------------
documents = []

if uploaded_files:
    temp_dir = "./temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

   # SELECT THE RIGHT LOADER
        if file.name.endswith(".pdf"):
            loader = PyPDFDirectoryLoader(temp_dir)
        else:
            # For .txt or .json, we load it as raw text
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path)
    docs = loader.load()
    documents.extend(docs)
    
    st.sidebar.success(f"✅ Loaded {len(documents)} pages!")

# -------------------------
# 7. Build RAG Chain
# -------------------------
conversation_rag_chain = None 

if documents:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,    
        chunk_overlap=2000,
        separators=["},", "\n\n"]   
    )
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    contextualise_q_system_prompt = (
        "Given a chat history and the latest user question which may reference "
        "the chat history, rewrite the question to be fully standalone. "
        "Do NOT answer it."
    )
    
    contextualise_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualise_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualise_q_prompt
    )

    qa_system_prompt = """You are a Senior QA Automation Engineer.
    Your task is to generate production-grade automation scripts based on the provided context.
    
    GUIDELINES:
    1. Analyze the Request (HTTP Method, URL, Body).
    2. Create POJO Classes if needed.
    3. Write TestNG/RestAssured tests.
    4. Include Negative Scenarios.
    5. If context is missing, state it clearly. Do not guess.

    CONTEXT:
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_message_key="answer"
    )

# -------------------------
# 8. Chat Interface (View History)
# -------------------------

# Display History FIRST so it stays on screen after refresh
if session_id in st.session_state.store:
    history = st.session_state.store[session_id].messages
    for message in history:
        if message.type == "human":
            with st.chat_message("user"):
                st.write(message.content)
        elif message.type == "ai":
            with st.chat_message("assistant"):
                st.write(message.content)

# Use st.chat_input (Only sends on Enter)
user_question = st.chat_input("Ask a question about your PDFs...")

if user_question:
    # Render user message instantly
    with st.chat_message("user"):
        st.write(user_question)

    if conversation_rag_chain is None:
        st.error("⚠️ Please upload a PDF first!")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = conversation_rag_chain.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": session_id}}
            )
            st.write(response["answer"])