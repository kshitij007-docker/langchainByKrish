## Load the properties file

import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## Langsmith Tracking

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']=os.getenv("LANGCHAIN_PROJECT")

## Prompt Template

prompt= ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant.Please respond to the question asked"),
    ("user","Question: {question}")
])

## Streamlit framework to create the web app

st.title("langchain demo with Ollama LLM model gemma3:1b")
input_text =st.text_input("Enter your first question here:")

## call ollama gemma3:1b 

llm =Ollama(model="gemma3:1b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
   st.write(chain.invoke({"question":input_text}))