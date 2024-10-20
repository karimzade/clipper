import streamlit as st
from clipper import (
    vectorizer,
    chunk_text,
    create_llm,
    get_transcript_data,
    search,
    get_llm_response,
    process_uploaded_video,
    transcribe_audio,
)

st.subheader("Clipper")
model = "llama3.1:8b"

summary_and_step_by_step_prompt = (
    "Write a concise summary (60-70 words) capturing the key points, structure, methods, tools, or technical concepts discussed. "
    "Follow with a detailed, step-by-step breakdown of the main flow in a bullet point format. "
    "Ensure both the Summary and Step-by-Step Explanation highlight critical topics and transitions seamlessly. Complete the template below.\n\n"
    "Summary\n"
    "XXX. YYY. ZZZ.\n\n"
    "Step-by-Step Explanation\n"
    "X\n"
    "- y\n"
    "- z\n\n"
    "M\n"
    "- n\n"
    "- b"
)

option = st.radio("Select Input Option:", ("Input URL", "Upload Video File"))

if option:
    st.session_state.clear()

video_file = None
url = None
texts = None

if option == "Input URL":
    url = st.text_input("Input URL")
elif option == "Upload Video File":
    video_file = st.file_uploader(
        "Upload Video File", type=["mp4", "mov", "avi", "mkv"]
    )

if url or video_file:
    if "texts" not in st.session_state:
        if url:
            texts = get_transcript_data(url)
        elif video_file:
            audio_path = process_uploaded_video(video_file)
            texts = transcribe_audio(audio_path)

        if texts:
            chunked_texts = chunk_text(texts)
            vectordb = vectorizer(texts)

            st.session_state.vectordb = vectordb
            st.session_state.chunked_texts = chunked_texts
            st.session_state.chain = create_llm(model)
            st.session_state.texts = texts

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question:")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectordb" in st.session_state and "chunked_texts" in st.session_state:
        match_docs = search(prompt, st.session_state.vectordb)
        response = get_llm_response(prompt, match_docs, st.session_state.chain)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        st.error("Please upload a video or input a URL to generate a summary first.")
else:
    if "texts" in st.session_state:
        response = get_llm_response(
            summary_and_step_by_step_prompt,
            st.session_state.chunked_texts,
            st.session_state.chain,
        )

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
