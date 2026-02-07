## RAG Q&A with pdf uploads including chat history

import streamlit as st
from langchain.chains import create_history_aware_retriever ,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.runnables import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN") ## Load HF token from .env file
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## set up streamlit app

st.title("RAG Q&A with PDF uploads including Chat History")
st.write("Upload Pdf's and chat 1 their content")

## Input the GROQ API key

api_key=st.text_input("Enter your GROQ API Key",type="password")

## Check if Groq API key is provided

if api_key:
    llm=ChatGroq(api_key=api_key,model_name="llama-3.1-8b-instant")

    ## Chat history
    session_id=st.text_input("Session ID",value="default_session") ## Session ID for chat history

  ## statefully manage chat history
    if 'store' not in st.session_state:
      st.session_state.store={}

    uploaded_files=st.file_uploader("Upload PDF files",type=["pdf"],accept_multiple_files=True)

    ## Process uploaded files
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as f:
            f.write(uploaded_file.getvalue())
            file_name=uploaded_file.name

        loader=PyPDFDirectoryLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)

        ## split documents into chunks and create vectorstore than retriever
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=20)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()


        contextualise_q_system_prompt= (
        "Given a chat history and the latest user question",
        "which might reference context in that chat history,"
        "formulate a standalone question which can be understood",
        "without the chat history. DO NOT answer the question",
        "just reformulate it if needed and otherwise return it as is",)

        contextualise_q_prompt=ChatPromptTemplate.from_messages([
        ("system",contextualise_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")])

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualise_q_prompt)

        ## Answer question

        system_prompt=("You are a helpful assistant.",
                    "Use the following context to answer the users question.",
                    "If you don't know the answer, just say that you don't know,",
                    "Use three sentences maximum."
                    
                    "\n\n"
                    "{context}"
                    )
        qa_prompt=ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")])

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
                return st.session_state.store[session_id]

        conversation_rag_chain=RunnableWithMessageHistory(rag_chain,get_session_history,input_messages_key="input",history_messages_key="chathistory",
                                                        output_message_key="output")
        
        user_input=st.text_input("Your Question:","")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversation_rag_chain.invoke({"input":user_input},config={"configurable":{"session_id":session_id}
                                                                                ## constructs a key "abc123" in a 'store'.
                                                                                })
            st.write(st.session_state.store)
            st.success("Assistant:",response['answer'])
            st.write("Chat History:",session_history.messages)

else:
    st.warning("Please enter your GROQ API key to proceed.")



 