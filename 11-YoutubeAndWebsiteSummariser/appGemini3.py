import sys
import os
import streamlit as st

# --- CRITICAL CONFIGURATION ---
# Get the absolute path to the 'libs' folder inside your project
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, "libs")

# Tell Python: "Look in 'libs' FIRST, before looking at the computer's system files"
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
# ------------------------------

import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document

# Now import the library. It SHOULD pick up the version from 'libs'
try:
    import youtube_transcript_api
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError as e:
    st.error(f"Failed to import library from {libs_path}. Error: {e}")
    st.stop()

## streamlit app
st.set_page_config(page_title="Langchain: Summarize", page_icon=":pencil:")
st.title(" :pencil: Langchain: Summarize text")

# --- DEBUG MESSAGE (You can remove this later) ---
# This proves if we are using the correct file
st.success(f"Library loaded from: {youtube_transcript_api.__file__}")
# -------------------------------------------------

with st.sidebar:
    groq_api_key = st.text_input("Enter your GROQ API Key", value="", type="password")
generic_url = st.text_input("URL", label_visibility="collapsed")

## --- CUSTOM LOADER ---
## --- UPDATED CUSTOM LOADER ---
## --- MODERN CUSTOM LOADER ---
def get_transcript_text(url):
    try:
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        else:
            video_id = url.split("v=")[-1].split("&")[0]

        # 1. Get the list of ALL available transcripts (The Modern Way)
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # 2. Filter: Ask for English ('en') OR Hindi ('hi')
        # This automatically prefers Manual captions, but falls back to Generated if needed.
        transcript = transcript_list.find_transcript(['en', 'hi'])
        
        # 3. Fetch the actual text data
        transcript_data = transcript.fetch()
        
        # 4. Combine into a single string
        text = " ".join([item['text'] for item in transcript_data])
        
        return [Document(page_content=text)]
        
    except Exception as e:
        raise Exception(f"Transcript Error: {e}")
## -----------------------------
## -----------------------------

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide inputs")
    elif not validators.url(generic_url):
        st.error("Invalid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    docs = get_transcript_text(generic_url)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False)
                    docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                st.success(chain.run(docs))
        except Exception as e:
            st.exception(f"Error: {e}")