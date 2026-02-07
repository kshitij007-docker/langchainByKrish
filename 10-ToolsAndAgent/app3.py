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
st.set_page_config(page_title="Debug Failures", page_icon="fire")
st.title(":fire: Failure Debugging Copilot")
st.caption("Paste your logs. I search StackOverflow, GitHub, and Official Docs for fixes.")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

# --- TOOL SETUP ---
# num_results=4 gives enough variety (e.g., 1 SO link, 1 GitHub link, 2 Docs links).
search = DuckDuckGoSearchResults(name="Search", num_results=4)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am your Debugging Copilot. Paste your error log below."
        }
    ]

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- MAIN LOGIC ---
if prompt := st.chat_input(placeholder="Paste error logs here (e.g., 500 Internal Server Error)"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.info("Please add your Groq API Key.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True
    )

    tools = [search]

    # --- THE "HIERARCHY OF TRUST" PROMPT ---
    # This instructs the AI on *how* to value the search results it finds.
    sys_prompt = """You are an expert Software Reliability Engineer.
    
    INSTRUCTIONS:
    1. ANALYZE the error log provided by the user.
    2. SEARCH the web for the error message or code.
    3. EVALUATE findings using this 'Hierarchy of Trust':
       - **Tier 1 (Best):** Official Documentation (e.g., docs.python.org, aws.amazon.com, react.dev).
       - **Tier 2 (Good):** High-quality community threads (StackOverflow, GitHub Issues).
       - **Tier 3 (Acceptable):** Technical blogs (Medium, Dev.to) if Tier 1/2 are missing.
       - **Ignore:** Generic marketing sites or SEO spam.
       
    4. OUTPUT FORMAT:
       - **Analysis:** Briefly explain what the error means.
       - **Probable Cause:** Why it happened.
       - **Recommended Solution:** Step-by-step fix.
       - **References:** List the URL of the most useful source you found.
    """

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
       ## agent_kwargs={'prefix': sys_prompt},
        max_iterations=5,                # Allow enough turns to read docs
      ##  early_stopping_method="generate"
    )

## Show assistant response with streaming in chat bubble

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # We run the agent. The StreamlitCallbackHandler allows you to SEE 
        # which URLs it is visiting in real-time.
        full_prompt = f"{sys_prompt}\n\nUSER ERROR LOG:\n{prompt}"
        response = search_agent.run(prompt, callbacks=[st_cb])
        
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})