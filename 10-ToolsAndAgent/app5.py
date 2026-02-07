import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="SecureLog Analyst", page_icon="🛡️")
st.title("🛡️" "SecureLog Analyst")
st.caption("Paste your logs. I search the web using your local model.")

## Sidebar for settings
st.sidebar.title("Settings")
model_id = st.sidebar.text_input("Ollama Model Name", value="gemma3:1b")

# --- CHANGE 1: RESET BUTTON ---
# Allows user to clear memory and start fresh
if st.sidebar.button("🗑️ Reset Conversation"):
    st.session_state.messages = []
    st.session_state.error_log = ""
    st.rerun()

search = DuckDuckGoSearchResults(name="Search", num_results=3)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Paste your error log below."}]

# --- CHANGE 2: MEMORY VARIABLE ---
if "error_log" not in st.session_state:
    st.session_state["error_log"] = ""

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- MAIN LOGIC ---
if prompt := st.chat_input(placeholder="Paste logs OR ask a follow-up..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatOllama(
        model=model_id,
        temperature=0,
        keep_alive="5m"
    )
    tools = [search]

    # --- CHANGE 3: CONTEXT AWARENESS LOGIC ---
    # Check if we already have the logs saved
    if not st.session_state.error_log:
        # CASE A: User just pasted the logs
        st.session_state.error_log = prompt
        
        
        # Bumped to 5000 (Ollama can handle this).
        truncated_prompt = prompt[:5000] 
        full_input = f"USER ERROR LOG:\n{truncated_prompt}"
        
        # PROMPT: Explicitly ask for ALL errors
        sys_prompt = sys_prompt = """You are an expert SRE. 
        
        MISSION:
        1. Scan entire Log and Identify EVERY distinct error in the log (look for 'Error', 'Failed', 'Denied').
        2. SEARCH for fixes using the Search tool.
        3. EXTRACT the URLs from the search results. You MUST cite them.
        
        CRITICAL OUTPUT RULES:
        - You MUST end your turn by outputting the "Final Answer".
        - The content of the "Final Answer" MUST be the Markdown table below.
        - Do not invent links. Only use links found in the 'Observation'.
        
        REQUIRED FINAL ANSWER FORMAT:
        
        Final Answer:
        **Total Errors Found:** [Exact Count of number of errors]

        **Analysis Table:**
        | Error | Solution | Probable Cause |
        | :--- | :--- | :--- |
        | [Error 1 Name] | [Fix Solution 1] | [Cause] |
        | [Error 2 Name] | [Fix Solution 2 ] | [Cause] |

        **References:** [Copy URLs from search results.Provide URLs used for searching the result.]
        """
        
    else:
        # CASE B: Follow-up question
        # We combine the saved log with the new question
        full_input = f"""
        CONTEXT (Existing Logs): 
        {st.session_state.error_log[:4000]}...
        
        USER FOLLOW-UP QUESTION: 
        {prompt}
        """
        sys_prompt = "You are a helpful assistant. Answer the user's follow-up question based on the error logs provided in context."

    # Initialize agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors="You are missing the 'Final Answer:' prefix. Please retry and start your response with 'Final Answer:' followed by the table and References",
        max_iterations=8,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Combine instructions with input This line glues them together into one long message so the AI receives both the Order and the Ingredients at the same time.
        final_prompt = f"{sys_prompt}\n\nINPUT:\n{full_input}" 
        
        try:
            response = search_agent.run(final_prompt, callbacks=[st_cb])
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Tip: Ensure Ollama is running.")