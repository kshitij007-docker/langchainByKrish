import streamlit as st

# 1. Setup Page Config FIRST to avoid Streamlit errors
st.set_page_config(page_title="QA Super-Copilot", page_icon="🕵️")
st.title("🕵️ QA Super-Copilot: Code, Docs & Video")

# 2. Safe Imports
try:
    from langchain_groq import ChatGroq
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.utilities import StackExchangeAPIWrapper
    from langchain_community.tools import YouTubeSearchTool
    from langchain.agents import initialize_agent, AgentType, Tool
    from langchain.callbacks import StreamlitCallbackHandler
except ImportError as e:
    st.error(f"❌ Missing Dependencies. Please look at your terminal or install: {e}")
    st.stop()

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

# --- TOOL 1: StackOverflow ---
try:
    stack_wrapper = StackExchangeAPIWrapper()
    stack_tool = Tool(
        name="StackOverflow",
        func=stack_wrapper.run,
        description="Useful for finding specific error messages and developer discussions."
    )
except Exception as e:
    # Use st.warning so the app doesn't crash completely, just disables the tool
    st.warning(f"⚠️ StackOverflow tool disabled. (Missing 'stackapi'?): {e}")
    stack_tool = None

# --- TOOL 2: GitHub (via DuckDuckGo) ---
try:
    ddg_search = DuckDuckGoSearchResults()
    def github_search_func(query):
        return ddg_search.run(f"site:github.com {query}")

    github_tool = Tool(
        name="GitHub_Search",
        func=github_search_func,
        description="Useful for finding code snippets, repo issues, and bug reports."
    )
except Exception as e:
    st.warning(f"⚠️ GitHub/DuckDuckGo tool disabled. (Missing 'duckduckgo-search'?): {e}")
    github_tool = None

# --- TOOL 3: YouTube Search ---
try:
    youtube_tool = YouTubeSearchTool()
except Exception as e:
    st.warning(f"⚠️ YouTube tool disabled. (Missing 'youtube-search'?): {e}")
    youtube_tool = None

# Filter out None tools
tools = [t for t in [stack_tool, github_tool, youtube_tool] if t is not None]

if not tools:
    st.error("❌ No tools are available. Please install the required libraries.")
    st.stop()

# --- CHAT UI ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "I can help you debug errors (StackOverflow/GitHub) or find setup tutorials (YouTube). What do you need?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ex: How to install Appium on Windows tutorial"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.info("Please enter your Groq API Key in the sidebar.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.1-8b-instant", 
        streaming=True
    )

    system_prompt = """
    You are an expert QA Technical Lead.
    ROUTING LOGIC:
    1. **ERROR LOGS/BUGS:** Use 'StackOverflow' or 'GitHub_Search'.
    2. **TUTORIALS/GUIDES:** If user asks for guides/tutorials/how-to, use 'youtube_search'.
    3. **GENERAL:** Use your own knowledge.
    """

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"prefix": system_prompt}
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        try:
            response = agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred during execution: {e}")