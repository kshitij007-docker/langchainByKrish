import streamlit as st ## used to build the web UI
import os ## used to read environment variables (API keys)
from langchain_groq import ChatGroq ## lets you use Groq LLMs with LangChain
from langchain_community.embeddings import OllamaEmbeddings ## creates embeddings using a local Ollama model (on your machine)
from langchain.text_splitter import RecursiveCharacterTextSplitter ## Splits long PDF text into smaller chunks (1000 characters in your case)
from langchain.chains.combine_documents import create_stuff_documents_chain ## This takes a) LLM b) your prompt c) the retrieved documents and stuffs them together before asking the model to answer
from langchain_core.prompts import ChatPromptTemplate ## Lets you create prompts with placeholders like {input} and {context}
from langchain.chains import create_retrieval_chain ## It combines retriever and LLM chain. This our main RAG pipeling
from langchain_community.vectorstores import FAISS ## FAISS stores vector embeddings for fast similarity search
from langchain_community.document_loaders import PyPDFDirectoryLoader ## Loads all PDF files from a folder automatically

from dotenv import load_dotenv ## Loads variables from .env (like your Groq API key)

load_dotenv()

## Load the GROQ API

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY") ## Ensures your API key is available for Groq
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

groq_api_key=os.getenv("GROQ_API_KEY") ## Fetch the Groq API key from environment variables and store it in a variable

llm = ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant") ## Creates a Groq LLM object using the fast llama-3.1-8b-instant model

## Define chat prompt template.This tells llm only answer from the retrieved context.Insert {context}- PDF chunks , {input} - user query

prompt=ChatPromptTemplate.from_template (

    """
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>

Question:{input}
"""
)

## create vector embeddings

def create_vector_embedding(): ## This creates a function that will prepare your RAG database (embeddings + FAISS).
    if "vectors" not in st.session_state: ## If the user hasn’t created the vector database yet, then run the following steps. (Streamlit’s session_state keeps data in memory while the app runs.)
        st.session_state.embeddings=OllamaEmbeddings(model="nomic-embed-text:latest") ## This loads the Ollama embedding model (nomic-embed-text) that will convert text → numbers.
        st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## Data ingestion --> This points to your folder research_papers/ that contains your PDFs.
        st.session_state.docs=st.session_state.loader.load() ## Document loader --> This loads all PDF pages into a list of documents.
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20) ## Text splitter --> This splits long PDF text into smaller chunks of 1000 characters with 20 characters overlap.
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) ## Takes only first 50 documents → splits them → stores chunks.
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) ## Converts each chunk into embeddings → stores them in FAISS vector database. --> Stores it in session_state for later use.Now RAG retriever have something to search!

st.title("RAG document Q&A with GROQ and Ollama Embedding model nomic-embed-text:latest") ## Title of the web app

user_prompt=st.text_input("Enter your query from the research paper")

## Create a button named as Document Embedding.When the user clicks the button:create_vector_embedding() runs -->After creation → display message: “Vector database is ready”
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vecotr database is ready")

import time

if user_prompt: ## If user typed something in the text box…
    if "vectors" not in st.session_state: ## If the user forgot to click the "Document Embedding" button → show error.
        st.error("Please click 'Document Embedding' first to create the vector database.")
    else: 
        document_chain = create_stuff_documents_chain(llm, prompt) ## Creates a chain that can pass documents + question into the LLM.
        retriever = st.session_state.vectors.as_retriever() ## This allows searching inside FAISS based on similarity by converting FAISS into a retriever object.
        retrieval_chain = create_retrieval_chain(retriever, document_chain) ## Combine retriever + LLM chain → RAG pipeline. This create : retrieve relevant chunks ---> pass them with the prompt -->get answer from LLM

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt}) ## Searches FAISS for relevant chunks --> Passes them to LLM --> Returns answer + context chunks
        print(f"Response time: {time.process_time() - start}") ## Print how fast it was

        st.write(response['answer']) ## Show answer to user

        with st.expander("Document similarity search"): ##Show retrieved documents in an expandable section
            for i, doc in enumerate(response['context']):  ## Shows the text chunks that were used to generate the final answer. Loop through each retrieved document chunk.
                st.write(doc.page_content)
                st.write("---------------------------")


