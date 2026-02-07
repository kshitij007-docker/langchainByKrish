import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith tracking

os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY') ## Load langchain api key from .env file

os.environ['LANGCHAIN_TRACING_V2']='true' ## Enable langchain tracing v2
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OPENAI" ## Project name in langchain studio

## Prompt template

prompt= ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Please response to the user queries"),
    ("user","Question:{question}")
])

def generate_response(question,api_key,llm,temperature,max_tokens):

    openai.api_key=api_key
    llm=ChatOpenAI(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer
## Title of the app
st.title("Q&A Chatbot with OPENAI")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your OpenAI API Key",type="password")

## Dropdown to select various open ai models

llm=st.sidebar.selectbox("Select OpenAI Model",["gpt-5.1","gpt-4","o4-mini"])

## Adjust response parameters. Temperature means creativity of the response it ranges from 0 to 1 and max tokens is the maximum length of the response
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7) ## 0.7 value will be selected by default
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main interface

st.write("Go Ahead and ask any question!")
user_input=st.text_input("Your Question:","")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get started.")