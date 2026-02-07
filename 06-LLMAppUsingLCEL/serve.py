## Deploy Langserve Runnable and chain as an API using FastAPI

from fastapi import FastAPI ## Import FastAPI for creating the API server.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes ## Import add_routes to add Langchain routes to FastAPI.Here add_routes is used to add the chain as an endpoint.
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model=ChatGroq(model="meta-llama/llama-guard-4-12b",groq_api_key=groq_api_key) ## Initialize the GROQ chat with the desired model and API key.
model ## Display the model object to confirm successful initialization.

# 1. Create Prompt Template

generic_template="Translate the following into {language}" 

prompt_template=ChatPromptTemplate.from_messages([ 
    ("system",generic_template),("user","{text}")]) ## from_messages will be used to create a chat prompt template from a list of messages.This messages will come from user and system.

parser=StrOutputParser()

## 2. Create Chain. Chain is used to link the prompt template, model, and output parser together.so that they can work in sequence to process inputs and generate outputs.

chain=prompt_template | model | parser

## App Definition. It means we are creating an instance of FastAPI with the specified title, version, and description.
app=FastAPI(title="Langchain Server" ,
            version="0.1",
            description="A simple API server using Langchain runnable interfaces")

## Adding chain routes. It means we are adding the chain as an endpoint to the FastAPI app at the specified path "/chain".For example, if the server is running locally on port 8000, the chain can be accessed at http://localhost:8000/chain.
add_routes(
    app,
    chain,
    path="/chain"
)
## Run the FastAPI app using Uvicorn when the script is executed directly.Uvicorn is an ASGI server for running FastAPI applications.
import uvicorn
if __name__ =="__main__": ## This line checks if the script is being run directly (not imported as a module).name 
    uvicorn.run(app, host="localhost",port=8000)

