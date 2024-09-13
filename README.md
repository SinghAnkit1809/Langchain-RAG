# Interactive ChatGroq with LangChain Demo

This is a Streamlit-based web application that demonstrates the integration of LangChain's `ChatGroq` language model with real-time TensorFlow search capabilities. The app enables users to ask questions about TensorFlow and receive interactive responses, while maintaining chat context throughout a session.

## Features

- **Real-time TensorFlow Search**: Ask any TensorFlow-related questions, and the app retrieves relevant information from the web.
- **Contextual Chat**: The chat system remembers your conversation history for the duration of the session, allowing for smooth and coherent dialogue.
- **Interactive UI**: Built with Streamlit, the interface provides a simple, user-friendly experience.
- **Session Management**: Chat history is stored for the session and can be cleared or reset as needed.

## How It Works

1. **Language Model**: The application uses LangChain's `ChatGroq` language model, initialized with a custom API key.
2. **Web-Based Document Loader**: TensorFlow-related content is loaded from a web page and split into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embedding and Search**: The content is embedded using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model, stored in a FAISS vector database, and retrieved using a search tool.
4. **OpenAI Tools Agent**: A custom agent is created to retrieve relevant information based on user queries and respond accordingly.

## Requirements

- Python 3.9 or later
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
