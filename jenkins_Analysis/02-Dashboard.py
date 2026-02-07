import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Jenkins Failure Copilot", layout="wide")

st.title("🛡️ AI-Assisted Build Failure Dashboard")
st.markdown("*A lightweight tool to analyze Jenkins failures for QA & Engineering Managers.*")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Select Model", ["gemma3:1b", "llama3", "mistral"], index=0)
    st.info("Ensure Ollama is running locally.")

# --- Main Interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Build Logs")
    # Option to upload file or paste text
    uploaded_file = st.file_uploader("Upload Jenkins Log File (.txt, .log)", type=['txt', 'log'])
    log_input = st.text_area("Or paste log snippet here:", height=300)

    analyze_btn = st.button("🚀 Analyze Failure", type="primary")

# --- Logic ---
if analyze_btn:
    log_text = ""
    
    # Determine source of log
    if uploaded_file is not None:
        log_text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif log_input:
        log_text = log_input
    
    if not log_text:
        st.error("Please upload a file or paste logs to proceed.")
    else:
        # Keep it bounded
        log_text = log_text[-6000:]
        
        with col2:
            st.subheader("2. AI Analysis Report")
            with st.spinner(f"Consulting {model_name} for root cause..."):
                try:
                    # Initialize LLM
                    llm = ChatOllama(model=model_name, temperature=0)

                    # Enhanced Manager-Friendly Prompt
                    prompt = f"""
                    You are a Senior Site Reliability Engineer reporting to a non-technical Manager.
                    
                    TASK: Analyze this Jenkins build log.
                    
                    OUTPUT FORMAT (Markdown):
                    
                    ### 🚨 Executive Summary
                    * **Severity:** [Low / Medium / Critical]
                    * **Impact:** [One line summary of what stopped]
                    * **Estimated Fix Effort:** [Low/High]
                    
                    ### 🛠️ Technical Root Cause
                    (Explain clearly, evidence-based only)
                    
                    ### 💡 Recommended Action
                    (Generic next step)
                    
                    LOG DATA:
                    {log_text}
                    """
                    
                    response = llm.invoke(prompt)
                    st.success("Analysis Complete")
                    st.markdown(response.content)
                    
                except Exception as e:
                    st.error(f"Analysis Failed: {str(e)}")