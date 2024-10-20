import os
import subprocess
import whisper
import streamlit as st
import tempfile
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.chains.question_answering import load_qa_chain
import re
from langchain.schema import Document
from pytubefix import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from pytubefix.cli import on_progress


def extract_video_id(link):
    """
    Extract the video ID from a YouTube link.

    Args:
        link (str): The YouTube video URL.

    Returns:
        str: The extracted video ID, or None if no ID was found.
    """
    patterns = [
        r"youtu\.be/([a-zA-Z0-9_-]+)",
        r"watch\?v=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    return None


def convert_video_to_audio(video_path):
    """
    Convert a video file to an audio file in MP3 format.

    Args:
        video_path (str): The path to the video file.

    Returns:
        str: The path to the converted audio file.
    """
    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    subprocess.run(["ffmpeg", "-i", video_path, audio_path])
    return audio_path


def transcribe_audio(path):
    """
    Transcribe the audio file using Whisper ASR model.

    Args:
        path (str): The path to the audio file.

    Returns:
        str: The transcribed text from the audio file, or None if an error occurs.
    """
    if path is None:
        print("Error: Invalid audio file path.")
        return None
    model = whisper.load_model("base")
    print("MP3 file is being transcribed...")
    result = model.transcribe(path)
    print("MP3 file has been transcribed.")
    return result["text"]


def download_transcript_data(url):
    """
    Download the transcript data for a given YouTube video URL.

    Args:
        url (str): The YouTube video URL.

    Returns:
        str: The combined transcript text of the video.
    """
    video_id = extract_video_id(url)
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    video_title = yt.title
    st.write(f"Video Title: {video_title}")
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_data = []
    for entry in transcript:
        start_time = entry["start"]
        text = entry["text"]
        transcript_data.append({"start": start_time, "text": text})
    combined_text = " ".join([entry["text"] for entry in transcript_data])
    return combined_text


def download_video_from_youtube(url):
    """
    Download a YouTube video and return its local path and title.

    Args:
        url (str): The YouTube video URL.

    Returns:
        Tuple[str, str]: The local path to the downloaded video and its title.
    """
    yt = YouTube(url, on_progress_callback=on_progress)
    video_title = yt.title.replace(" ", "_")
    video_stream = yt.streams.get_highest_resolution()
    video_path = video_stream.download(filename=f"{video_title}.mp4")
    return video_path, yt.title


def get_transcript_data(url):
    """
    Retrieve the transcript data for a YouTube video, downloading and transcribing if necessary.

    Args:
        url (str): The YouTube video URL.

    Returns:
        str: The combined transcript text of the video.
    """
    try:
        texts = download_transcript_data(url)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(
            f"Transcript unavailable, error: {e}. Downloading video and transcribing audio..."
        )
        video_path, video_title = download_video_from_youtube(url)
        audio_path = convert_video_to_audio(video_path)
        texts = transcribe_audio(audio_path)
        st.write(f"Video Title: {video_title}")
    return texts


def chunk_text(text):
    """
    Split the input text into manageable chunks for processing.

    Args:
        text (str): The text to be split.

    Returns:
        List[Document]: A list of Document objects containing the text chunks.
    """
    documents = [Document(page_content=text)]
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents


def vectorizer(text):
    """
    Create a vector database from the input text using embeddings.

    Args:
        text (str): The text to be vectorized.

    Returns:
        Chroma: The vector database containing the text embeddings.
    """
    chunked_documents = chunk_text(text)
    client = chromadb.Client()
    if not client.list_collections():
        client.create_collection("consent_collection")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    vectordb.persist()
    return vectordb


def create_llm(model_name):
    """
    Create a language model chain for question answering.

    Args:
        model_name (str): The name of the language model to be used.

    Returns:
        Chain: The loaded question answering chain using the specified model.
    """
    llm = OllamaLLM(model=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def search(query, vectordb):
    """
    Search for documents in the vector database that match the query.

    Args:
        query (str): The search query.
        vectordb (Chroma): The vector database to search.

    Returns:
        List[Document]: A list of documents that match the query.
    """
    matching_docs = vectordb.similarity_search(query, k=10)
    return matching_docs


def get_llm_response(query, matching_docs, chain):
    """
    Get a response from the language model based on the query and matching documents.

    Args:
        query (str): The query to be processed.
        matching_docs (List[Document]): The documents relevant to the query.
        chain (Chain): The question answering chain.

    Returns:
        str: The response generated by the language model.
    """
    return chain.run(input_documents=matching_docs, question=query)


def process_uploaded_video(video_file):
    """
    Process an uploaded video file, converting it to audio format.

    Args:
        video_file (UploadedFile): The uploaded video file.

    Returns:
        str: The path to the converted audio file, or None if an error occurs.
    """
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_video_file:
            temp_video_file.write(video_file.read())
            temp_video_path = temp_video_file.name

        audio_path = os.path.splitext(temp_video_path)[0] + ".mp3"
        subprocess.run(["ffmpeg", "-i", temp_video_path, audio_path], check=True)

        return audio_path

    except Exception as e:
        st.error(f"An error occurred while processing the video: {e}")
        return None
