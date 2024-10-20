# Clipper

LLM-Powered YouTube Video Summarizer

## ✨ Features

- [X] Input URL and Upload Video Feature
- [X] Generate Summary and Step-by-Step Explanation Guide
- [X] Chat with LLM about Video
- [ ] Language Translation and Voiceover

## 🛠️ Tech Stack

- **Chatbot:** Llama 3.1 8b
- **Speech-to-Text Model:** OpenAI Whisper
- **Web UI:** Streamlit
- **Chain:** LangChain

## 🧰 Setup and Usage

### Prerequisites

- Python 3.8+
- [Git](https://git-scm.com/)
- [Ollama Llama 3.1 8b](https://ollama.com/library/llama3.1:8b)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/karimzade/clipper.git
   cd clipper
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate # For Windows: venv\\Scripts\\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app/streamlit.py 
   ```
