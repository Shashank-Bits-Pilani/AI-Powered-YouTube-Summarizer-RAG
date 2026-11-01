# AI-Powered YouTube Summarizer & Q&A Tool using RAG

An AI-powered application that summarizes YouTube videos and allows users to ask questions about their content using **Retrieval-Augmented Generation (RAG)**, **LangChain**, **FAISS**, and **Ollama (Llama 3.2)**.

The application provides an interactive interface built with **Gradio**, making it easier to extract useful information from YouTube videos without manually reviewing the entire content.

## Features

- **YouTube Transcript Extraction:** Retrieve available transcripts from YouTube videos.
- **AI-Powered Summarization:** Generate summaries using a locally running Llama 3.2 model.
- **Question Answering:** Ask questions about video content using RAG.
- **Semantic Search:** Retrieve relevant transcript sections using FAISS vector search.
- **Sentence Embeddings:** Convert transcript chunks into embeddings using SentenceTransformers.
- **Interactive Interface:** Access the application through a Gradio web interface.
- **Local LLM Inference:** Run the language model locally using Ollama.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| LangChain | LLM and RAG orchestration |
| FAISS | Vector similarity search |
| Ollama | Local LLM inference |
| Llama 3.2 | Text generation and question answering |
| SentenceTransformers | Text embeddings |
| Gradio | Web interface |
| YouTube Transcript API | Transcript extraction |

## How It Works

1. The user provides a YouTube video URL.
2. The application extracts the available video transcript.
3. The transcript is divided into smaller text chunks.
4. SentenceTransformers generates embeddings for those chunks.
5. FAISS indexes the embeddings for similarity search.
6. LangChain retrieves relevant transcript chunks for user questions.
7. Llama 3.2 generates responses using the retrieved context.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shashank-Bits-Pilani/AI-Powered-YouTube-Summarizer-RAG.git
cd AI-Powered-YouTube-Summarizer-RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv311
```

Activate it on Windows PowerShell:

```powershell
.\.venv311\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and configure Ollama

Install Ollama from https://ollama.com.

Download the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before starting the application.

### 5. Run the application

```bash
python ytbot.py
```

Open the local Gradio URL displayed in your terminal.

## Example Use Cases

- Summarizing educational YouTube lectures
- Extracting insights from technical tutorials
- Asking questions about video transcripts
- Reviewing research discussions and presentations
- Building AI-powered knowledge retrieval applications

## Future Improvements

- Support for multilingual video transcripts
- Improved summarization of long videos
- Timestamp-based answers
- Support for multiple YouTube videos
- Cloud deployment
- Enhanced retrieval and response evaluation

## Author

**Shashank Singh**

GitHub: https://github.com/Shashank-Bits-Pilani

## License

## License

This project is licensed under the **MIT License**.

Copyright © 2026 Shashank Singh.

See the [LICENSE](LICENSE) file for the full license terms.
