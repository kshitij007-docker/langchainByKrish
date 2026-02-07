# app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- helper ----------
def system_prompt(mode: str) -> str:
    if mode == "QA – Fix my test failure":
        return (
            "You are a senior QA automation engineer.\n"
            "User will paste test failures, logs, or stack traces.\n\n"
            "Your job:\n"
            "1. Explain the failure in simple, clear language.\n"
            "2. Identify the most likely root cause.\n"
            "3. Suggest 3 concrete fixes the QA can try now.\n"
            "4. Say whether this looks like a test issue, environment issue, or product bug.\n\n"
            "Be practical and specific. Avoid generic theory."
        )
    if mode == "QA Manager – Debug team failures":
        return (
            "You are a QA Manager assistant.\n"
            "User will paste failures from multiple tests or builds.\n\n"
            "Your job:\n"
            "1. Classify each failure (flaky test, environment, data, product bug, infra).\n"
            "2. Identify any patterns across failures.\n"
            "3. Recommend next owner (QA, Dev, Infra/DevOps).\n"
            "4. Suggest preventive actions to avoid repeat failures.\n\n"
            "Focus on delivery risk, efficiency, and team-level insights."
        )
    if mode == "Company – RCA for failures":
        return (
            "You are an incident RCA assistant.\n"
            "User will paste production or test failure details.\n\n"
            "Your job:\n"
            "1. Summarize the incident in business-friendly language.\n"
            "2. Identify primary and contributing technical causes.\n"
            "3. Suggest corrective and preventive actions (CAPA).\n"
            "4. Highlight risk if unresolved.\n\n"
            "Keep it concise and suitable for RCA / postmortem documents."
        )
    return "You are a helpful assistant for debugging software and test failures."

# ---------- tools (unchanged) ----------
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchResults(name="Search")
tools = [wiki, arxiv, search]

# ---------- UI ----------
st.set_page_config(page_title="QA Failure Debug Copilot", layout="wide")
st.title("QA Failure Debug Copilot")

st.sidebar.title("Settings")
# Prefer env var, but allow manual input
env_key = os.getenv("GROQ_API_KEY", "")
api_key = st.sidebar.text_input("Enter your GROQ API Key (optional)", type="password", value=env_key)

mode = st.sidebar.selectbox(
    "Who are you?",
    [
        "QA – Fix my test failure",
        "QA Manager – Debug team failures",
        "Company – RCA for failures"
    ],
)

# Helpful info area
with st.expander("How to use (quick)"):
    st.markdown(
        "- Paste a failing test log, stack trace or error message into the input box.\n"
        "- Choose mode (QA / QA Manager / Company) to change the response format.\n"
        "- If you don't have a GROQ key, the app will run in demo mode with sample output.\n"
        "- If the app errors contacting the LLM, you'll still see a readable error message here."
    )

# Initialize session messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hi — QA Failure Debug Copilot is ready.\n\n"
                "Paste your failing test logs, stack traces, or error messages and I will:\n"
                "- Explain what went wrong\n"
                "- Suggest likely root cause\n"
                "- Recommend concrete next steps\n\n"
                f"Current mode: {mode}"
            )
        }
    ]

# show history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input: prefer chat_input if available (newer Streamlit). Fallback to text_area+button
use_chat_input = hasattr(st, "chat_input")
if use_chat_input:
    user_input = st.chat_input(placeholder="Paste your error log / stack trace / failure details here...")
    submit_button = True if user_input else False
else:
    user_input = st.text_area("Paste your error log / stack trace / failure details here...", height=200)
    submit_button = st.button("Submit")

# If nothing provided, show a helpful sample and stop (so UI is not blank)
if not user_input and not submit_button:
    st.info("Paste a failure and submit. You can also run in demo mode without an API key.")
    st.stop()

# When user submits or chat_input filled
if user_input:
    # save user message and display
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # If no api_key, run demo response (so UI never blank)
    if not api_key:
        demo_resp = (
            "DEMO MODE (no API key supplied). Example analysis:\n\n"
            "1) Short explanation: The failing test shows a `NoSuchElementException` at `findElement` — likely locator mismatch or timing issue.\n"
            "2) Likely root cause: Locator changed or page load timing; test tries to interact before element exists.\n"
            "3) Quick fixes to try now:\n"
            "   - Add explicit WebDriverWait for element visibility before interacting.\n"
            "   - Verify locator with browser devtools; update to stable CSS/XPath.\n"
            "   - If intermittent, add retry logic or ensure page is fully loaded.\n"
            "4) Looks like: Test issue (most likely) — mark as flaky until root cause confirmed.\n\n"
            "Switch to a real GROQ API key in the sidebar to get live web-sourced suggestions."
        )
        st.session_state["messages"].append({"role": "assistant", "content": demo_resp})
        st.chat_message("assistant").write(demo_resp)
    else:
        # Try to call real agent but catch exceptions and show friendly message
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama-3.1-8b-instant",
                streaming=True,
            )

            search_agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
            )

            # Compose mode-specific prompt + user input
            sys_text = system_prompt(mode)
            full_input = sys_text + "\n\nUser failure details:\n" + user_input

            # Stream response if possible
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                # agent.run may throw; show error if it does
                response = search_agent.run(full_input, callbacks=[st_cb])
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.write(response)

        except Exception as e:
            # Friendly error info instead of blank page
            err_msg = (
                "Error running the LLM agent:\n\n"
                f"{repr(e)}\n\n"
                "Possible causes:\n"
                "- Invalid/expired API key\n"
                "- Network/connectivity problems\n"
                "- LangChain / Groq client mismatch versions\n\n"
                "Run in DEMO mode (remove API key) to see an example, or check the error above."
            )
            st.session_state["messages"].append({"role": "assistant", "content": err_msg})
            st.chat_message("assistant").write(err_msg)
