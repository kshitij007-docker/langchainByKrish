import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the app title
st.title("🐞 Failure Debugging Copilot")
st.caption("Paste your error logs below. I will find the cause and suggested solutions.")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

# --- TOOL SETUP ---
# For debugging, we remove Arxiv/Wiki. We strictly want web search (StackOverflow, Docs, Github).
search = DuckDuckGoSearchResults(name="Search")

# Initialize chat history if it does not exist
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am your Debugging Copilot. Please paste your error log, stack trace, or code snippet."
        }
    ]

# Display all previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- MAIN LOGIC ---
if prompt := st.chat_input(placeholder="Paste error logs here (e.g., ModuleNotFoundError: No module named 'xyz')"):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Check for API Key
    if not api_key:
        st.info("Please add your Groq API Key in the sidebar to continue.")
        st.stop()

    # 2. Initialize LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True
    )

    # 3. Define Tools
    tools = [search]

    # 4. Define Agent Persona (The "Brain" Tweak)
    # We give the agent a specific instruction to act as a troubleshooter.
    sys_prompt = """You are an expert Software Reliability Engineer and Debugging Assistant. 
    When a user pastes logs or error messages:
    1. Analyze the stack trace or error code.
    2. Use Search to find similar issues on StackOverflow or GitHub if you don't know the answer.
    3. Output the response in this format:
       - **Root Cause:** Explanation of why this happened.
       - **Solution:** Step-by-step fix.
    """

    # 5. Initialize Agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        agent_kwargs={'prefix': sys_prompt} # Inject the persona here
    )

    # 6. Generate Response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Run the agent
        # We pass the prompt directly to ensure the agent focuses on the immediate log
        response = search_agent.run(prompt, callbacks=[st_cb])
        
        st.write(response)

    # 7. Save Assistant Response
    st.session_state.messages.append({"role": "assistant", "content": response})