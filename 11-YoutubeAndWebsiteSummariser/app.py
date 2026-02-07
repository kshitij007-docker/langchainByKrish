import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


import sys
import subprocess

import youtube_transcript_api
import os



# # --- DIAGNOSTIC & AUTO-FIX BLOCK ---
# try:
#     import youtube_transcript_api
#     # If the library is too old, this line forces an upgrade
#     if not hasattr(youtube_transcript_api.YouTubeTranscriptApi, 'list_transcripts'):
#         st.warning("⚠️ Old library version detected. Upgrading now...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "youtube-transcript-api"])
#         st.success("✅ Upgrade complete! PLEASE STOP AND RESTART THIS APP.")
#         st.stop() # Stops execution here so you can restart
# except ImportError:
#     pass
# # -----------------------------------

## streamlit app

st.set_page_config(page_title="Langchain: Summarize text from URL/YouTube",page_icon=":pencil:")
st.title(" :pencil: Langchain: Summarize text from URL/YouTube")
st.subheader("Summarize URL")

# This prints the location of the file Python is actually using
st.write("I am loading the library from here:")
st.code(youtube_transcript_api.__file__)
## Get the GROQ API Key and URL to be summarized

with st.sidebar:
    groq_api_key=st.text_input("Enter your GROQ API Key",value="",type="password")
    
generic_url=st.text_input("URL",label_visibility="collapsed")



prompt_template="""
Provide a summary of the following content in 300 words in simple Words:
Content: {text}

"""
prompt =PromptTemplate(template=prompt_template,input_variables=["text"])
if st.button("Summarize the content from YT or website"):
    ## Validate all inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide all the inputs to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")
    
    else:
        try:
            with st.spinner("Waiting..."):

                ## Initialize Groq LLM. We can also use Gemma-7b-It
                llm=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)
                ## Loading website or YT video data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"})
                docs=loader.load()

                ## Chain for Summarization

                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")

                

