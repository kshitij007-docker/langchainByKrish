from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith tracking

os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY') ## Load langchain api key from .env file

os.environ['LANGCHAIN_TRACING_V2']='true' ## Enable langchain tracing v2
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OLLAMA" ## Project name in langchain studio

## Prompt template

prompt= ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Please response to the user queries"),
    ("user","Question:{question}")
])

def generate_response(question,temperature,max_tokens):

    llm=Ollama(model="gemma3:1b",temperature=temperature,num_predict=max_tokens) ## Num_predict is equivalent to max tokens in Ollama
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer

## Title of the app
st.title("Q&A Chatbot with OLLAMA")

## Sidebar for settings
st.sidebar.title("Settings")



## Adjust response parameters. Temperature means creativity of the response it ranges from 0 to 1 and max tokens is the maximum length of the response
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7) ## 0.7 value will be selected by default
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main interface

st.write("Go Ahead and ask any question!")
user_input=st.text_input("Your Question:","")

if user_input:
    response=generate_response(user_input,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get started.")