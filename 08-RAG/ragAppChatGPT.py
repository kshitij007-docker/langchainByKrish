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
# Load environment variables
# -------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN


# -------------------------
# Embeddings model
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# Streamlit UI
# -------------------------
st.title("📚 RAG Q&A with PDF Uploads + Chat History")
st.write("Upload PDFs and ask questions based on their content.")


api_key = st.text_input("Enter your GROQ API Key", type="password")

if not api_key:
    st.warning("Please enter your GROQ API Key to continue.")
    st.stop()


# Create LLM
llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")


# Chat session ID
session_id = st.text_input("Session ID", value="default_session")


# Store conversation histories
if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


# -------------------------
# PDF Upload + Processing
# -------------------------
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

documents = None

if uploaded_files:
    temp_dir = "./temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)

    # save files
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

    # Load PDFs
    loader = PyPDFDirectoryLoader(temp_dir)
    documents = loader.load()

    st.success(f"Loaded {len(documents)} pages from PDFs!")


# -------------------------
# Build RAG chain only after PDFs uploaded
# -------------------------
if documents:

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)

    # Vector store
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Prompts
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

    qa_system_prompt = (
        "You are a helpful assistant. Use the given context to answer the user’s question. "
        "If the answer is not in the context, say 'I don't know'. "
        "Keep the answer within 3 sentences.\n\n{context}"
    )

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

    # --------------------
    # USER INPUT
    # --------------------
    user_question = st.text_input("Your question:")

    if user_question:
        session_history = get_session_history(session_id)

        response = conversation_rag_chain.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}}
        )

        st.write("### 🤖 Assistant:")
        st.success(response["answer"])

        st.write("### Chat History:")
        st.write(session_history.messages)
