import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchResults
from langchain.agents import initialize_agent,AgentType
from langchain import hub
from langchain.callbacks import StreamlitCallbackHandler # Import Streamlit callback to stream LLM output live.It is a listener.It listens to the AI while it’s generating text.Every time new text comes, it sends it to Streamlit UI
import os
from dotenv import load_dotenv





## Wikipedia Tool

### Create an object that knows how to fetch data from Wikipedia
api_wrapper_wiki = WikipediaAPIWrapper(
    top_k_results=1,# Get only 1 best search result
    doc_content_chars_max=250 # Limit article text to 250 characters
)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki) # Create a Wikipedia search tool using the above wrapper
wiki.name # Get the internal name of the Wikipedia tool (used by LangChain agents to identify it)

##Arxiv tool

# Create an object that knows how to fetch research papers from Arxiv
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv) # Create an Arxiv search tool using the above wrapper
arxiv.name  ##Get the internal name of the Arxiv tool (used by LangChain)

search= DuckDuckGoSearchResults(name="Search") # # Create a DuckDuckGo search tool for general web search

# Set the app title shown on the Streamlit page

st.title("Langchain - Chat with search")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your GROQ API Key", type="password")

# Initialize chat history if it does not exist

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role" : "assistant", # Message sender (assistant)
            "content":"Hi,I am a chatbot who can search the web.How can I help you?"
        }
    ]
    # Display all previous messages from session state

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"]) #msg["role"] → Who sent the message (user or assistant). msg["content"] → The actual text message

# Take user input from chat box
## := is called walrus operator. It means “Get input AND store it in a variable at the same time"
## If user types something and presses Enter: Store it in prompt --> Run code inside if block

if prompt := st.chat_input(placeholder="What is machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt}) # Take what the user typed --> Save it in chat history --> Mark it as a "user" message
    st.chat_message("user").write(prompt) # Immediately show the user’s message on screen

# Initialize Groq LLM with streaming enabled

    llm=ChatGroq(groq_api_key=api_key,  # User-provided API key
                 model_name="llama-3.1-8b-instant", 
                 streaming=True # Enable token-by-token streaming

                 )

    ## Create list of tools.Combine all tools that agent can use
    tools=[wiki,arxiv,search]

    # Create the agent.Initialize an agent that decides which tool to use automatically
    search_agent=initialize_agent(
        tools,
        llm, #Give the agent a brain. The agent uses this LLM to Understand the question, Decide what tool to use,Write the final answer
        agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Reason + Act agent type. i.e. The agent thinks first, then acts by choosing a tool.
        handle_parsing_errors=True # Prevent crashes on wrong format.If the AI response is slightly malformed or confusing, don’t crash — try to recover.
        # with parsing errors agent tries to fix its own mistakes and app keeps running
    )
    ##search_agent = create_react_agent(llm, tools, prompt)
# Display assistant message container

    with st.chat_message("assistant"): ## Open a chat bubble that belongs to the assistant.Everything inside this block appears as message from the assistant and is aligned on assistant side like chatgpt replies
        st_cb= StreamlitCallbackHandler(st.container(),expand_new_thoughts=False) # # It watches the AI while it’s “typing”.Shows words gradually on the screen.This works only because streaming=True was enabled in the LLM.
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb]) # Ask the AI agent to generate a reply using the full chat history Here Agent reads all previous messages -->Thinks about the conversation --> Decides which tool to use -->Gets Information --> Starts Answering --> Streams the answer live to UI
        st.session_state.messages.append({"role":"assistant","content":response}) # Save the assistant’s final reply into memory.Hence,it doesn’t disappear when Streamlit refreshes and future answers remember past messages
        st.write(response)  # Even though the response was streamed live, this ensures the final completed answer is shown correctly.

        

    ## Get response from agent
    ##response=search_agent.run(prompt) # Send the user’s question (prompt) to the AI agent and get the answer.

    st.session_state.messages.append({"role":"assistant","content":response}) # Save the AI’s reply in memory.
    st.chat_message("assistant").write(response) # Show response in chat UI. Display the AI’s message in the chat UI.