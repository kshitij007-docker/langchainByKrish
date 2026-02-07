import sys
import os

# ---------------------------------------------------------
# Force Python to use local ./libs first (to avoid Anaconda conflicts)
# ---------------------------------------------------------
sys.path.insert(0, os.path.abspath("./libs"))

import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document

# Try to import youtube_transcript_api from ./libs
try:
    import youtube_transcript_api as yta_module
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    st.error("⚠️ youtube-transcript-api not found. Run:\n\npip install -t ./libs youtube-transcript-api==0.6.2")
    st.stop()

# ---------------------------------------------------------
# Optional debug – uncomment if you want to see what is imported
# ---------------------------------------------------------
# st.write("youtube_transcript_api loaded from:", getattr(yta_module, "__file__", "unknown"))
# st.write("YouTubeTranscriptApi attributes:", dir(YouTubeTranscriptApi))

# ---------------------------------------------------------
# Streamlit app config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Langchain: Summarize text from URL/YouTube",
    page_icon="✏️"
)

st.title("✏️ Langchain: Summarize text from URL / YouTube")
st.subheader("Paste any article or YouTube link and get a summary")

# ---------------------------------------------------------
# Sidebar – GROQ API key
# ---------------------------------------------------------
with st.sidebar:
    groq_api_key = st.text_input(
        "Enter your GROQ API Key",
        value="",
        type="password"
    )

# ---------------------------------------------------------
# URL input
# ---------------------------------------------------------
generic_url = st.text_input(
    "Enter URL (YouTube or Website)",
    placeholder="https://www.youtube.com/watch?v=... or https://example.com/article",
)


# ---------------------------------------------------------
# Helper: Extract YouTube Video ID
# ---------------------------------------------------------
def extract_youtube_video_id(url: str) -> str:
    """
    Extracts the video ID from youtube.com or youtu.be URLs.
    Raises ValueError if it can't find a valid ID.
    """
    if "youtu.be" in url:
        # e.g. https://youtu.be/VIDEO_ID?si=...
        video_id = url.split("/")[-1].split("?")[0].strip()
    elif "youtube.com" in url and "v=" in url:
        # e.g. https://www.youtube.com/watch?v=VIDEO_ID&ab_channel=...
        video_id = url.split("v=")[-1].split("&")[0].strip()
    else:
        raise ValueError("Not a valid YouTube URL format")

    if not video_id:
        raise ValueError("Could not parse video ID from URL")

    return video_id


# ---------------------------------------------------------
# Robust YouTube transcript loader
# ---------------------------------------------------------
def get_youtube_transcript_docs(url: str):
    """
    Returns a list[Document] containing the YouTube transcript text.
    Handles both get_transcript and list_transcripts variants.
    Raises a clear exception on failure.
    """
    try:
        video_id = extract_youtube_video_id(url)

        # If the class has get_transcript (most common)
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=["en", "en-US", "en-GB"]  # try English variants
            )

        # If only list_transcripts exists (some versions)
        elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            # Try to get an English transcript, else fallback to first available
            try:
                transcript = transcripts.find_transcript(["en", "en-US", "en-GB"])
            except Exception:
                transcript = next(iter(transcripts))
            transcript_list = transcript.fetch()

        else:
            # You imported something that is not the real YouTubeTranscriptApi class
            raise AttributeError(
                f"YouTubeTranscriptApi has neither 'get_transcript' nor 'list_transcripts'. "
                f"Current attributes: {dir(YouTubeTranscriptApi)}"
            )

        # Join transcript text
        transcript_text = " ".join(
            item.get("text", "") for item in transcript_list if item.get("text")
        ).strip()

        if not transcript_text:
            raise RuntimeError("Transcript loaded but text is empty.")

        return [Document(page_content=transcript_text)]

    except Exception as e:
        # Show detailed error in the UI and rethrow to be caught by outer try
        st.exception(f"Transcript error: {e}")
        raise Exception(f"Could not fetch transcript. Error: {e}")


# ---------------------------------------------------------
# Prompt template
# ---------------------------------------------------------
prompt_template = """
Provide a summary of the following content in about 300 words using very simple, easy-to-understand language:

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

# ---------------------------------------------------------
# Main button logic
# ---------------------------------------------------------
if st.button("Summarize the content from URL / YouTube"):
    # Basic validation
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both: GROQ API Key and a URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Summarizing..."):
                # Initialize LLM
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    groq_api_key=groq_api_key
                )

                # Detect YouTube vs normal webpage
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    docs = get_youtube_transcript_docs(generic_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/58.0.3029.110 Safari/537.3"
                            )
                        },
                    )
                    docs = loader.load()

                # Summarization chain
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )
                output_summary = chain.run(docs)

                st.success("Summary:")
                st.write(output_summary)

        except Exception as e:
            # Show full exception with traceback in Streamlit
            st.exception(f"Exception: {e}")
