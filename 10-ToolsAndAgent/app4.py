import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables (if you have other env vars)
load_dotenv()

# Set the app title
st.set_page_config(page_title="Local Debugger", page_icon="🦙")
st.title("🦙 Local Failure Debugging Copilot (Ollama)")
st.caption("Paste your logs. I search the web using your local gemma3:1b.")

## Sidebar for settings
st.sidebar.title("Settings")
# We no longer need an API Key, but we can let user pick the model name
model_id = st.sidebar.text_input("Ollama Model Name", value="gemma3:1b")

# --- TOOL SETUP ---
# num_results=3 gives enough variety without overwhelming the context
search = DuckDuckGoSearchResults(name="Search", num_results=3)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am running locally on Ollama. Paste your error log below."
        }
    ]

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- MAIN LOGIC ---
if prompt := st.chat_input(placeholder="Paste error logs here..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # CHANGE 1: Use ChatOllama instead of ChatGroq
    llm = ChatOllama(
        model=model_id,
        temperature=0,      # Keep it factual
        keep_alive="5m"     # Keeps the model loaded in RAM for 5 mins
    )

    tools = [search]

    # --- PROMPT SETUP ---
    sys_prompt = """You are an expert Software Reliability Engineer.

    1. ANALYZE the error log.
    2. SEARCH the web for fixes.
    3. EVALUATE findings.

    OUTPUT CONTENT:
    Your final answer must contain these sections:
    - **Analysis:** [Brief explanation in simple words]
    - **Probable Cause:** [Why it happened]
    - **Number of Errors:** [Give number of errors present in the logs]
    - **Recommended Solution:** [Code fix/steps]
    - **References:** [URL]
    """

    # Initialize the agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # Helper string to fix the agent if it gets stuck in a loop
        handle_parsing_errors="Check your output format. Do not just output text, you must use the Action format or 'Final Answer'.",
        max_iterations=5,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Truncate prompt to safe limits (approx 1000 chars)
        truncated_prompt = prompt[:1000]
        full_prompt = f"{sys_prompt}\n\nUSER ERROR LOG:\n{truncated_prompt}"
        
        try:
            response = search_agent.run(full_prompt, callbacks=[st_cb])
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Tip: Make sure Ollama is running (`ollama serve`)")