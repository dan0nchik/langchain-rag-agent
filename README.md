# Advanced RAG Pipeline with DeepEval Evaluation

This project implements a Retrieval Augmented Generation (RAG) pipeline using Langchain, with options for **both** standard vector search and GraphRAG. It evaluates the RAG pipeline's performance on a custom Q&A dataset using the DeepEval framework.

The dataset used for evaluation is here: https://github.com/docugami/KG-RAG-datasets/tree/main

## Features

*   Supports multiple LLM providers (OpenAI, Ollama).
*   Supports multiple embedding model providers (OpenAI, Ollama).
*   Implements both Standard RAG and GraphRAG retrieval strategies.
*   Processes PDF documents for the knowledge base.
*   Uses Chroma for vector storage with persistence options.
*   Evaluates the RAG pipeline using DeepEval with metrics like Contextual Precision, Recall, Relevancy, Answer Relevancy, and Faithfulness.
*   Comprehensive logging for traceability.

## Prerequisites

*   Python 3.9+
* Ollama installed or OpenAI API Key
*   [**uv**](https://github.com/astral-sh/uv): A fast Python package installer and resolver.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dan0nchik/langchain-rag-agent.git
    cd langchain-rag-agent
    ```

2.  **Install `uv` package manager:**
    Follow the official installation instructions for `uv` from [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation). For example:
    ```bash
    # On macOS and Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    Verify the installation:
    ```bash
    uv --version
    ```

3.  **Create and sync a virtual environment:**
    ```bash
    uv sync
    ```

## Configuration

Configuration for the RAG pipeline and evaluation is managed through `config.py` and a `.env` file for sensitive keys.

### 1. `config.py`

This file (`config.py`) contains various settings for the RAG pipeline, such as:
*   `LLM_PROVIDER`: "ollama" or "chatgpt".
*   `OLLAMA_LLM_MODEL`, `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_BASE_URL`.
*   `OPENAI_LLM_MODEL`, `OPENAI_EMBEDDING_MODEL`.
*   `RAG_TYPE`: "standard" or "graph".
*   `PDF_DIRECTORY_PATH`: Path to the directory containing your PDF files (default is "docs" from https://github.com/docugami/KG-RAG-datasets/tree/main).
*   `QNA_CSV_PATH`: Path to your CSV file with "Question" and "Answer" columns (default is eval_qna).
*   Chunking parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
*   Retriever parameters (`STANDARD_RETRIEVER_K`, GraphRAG strategy parameters).
*   ChromaDB settings (`CHROMA_PERSIST_DIRECTORY`, collection name logic).
*   DeepEval metric model provider (`DEEPEVAL_METRICS_MODEL_PROVIDER`).

Review and modify `config.py` to suit your specific setup and models.

### 2. `.env` File

Create a `.env` file in the root directory of the project to store sensitive information and API keys. **Do not commit this file to version control.**

Example `.env` file content:

```env
OPENAI_API_KEY="your_openai_api_key_here"

OLLAMA_BASE_URL="http://localhost:11434" # Only if different from default in config.py or if you want to override
```

### 3. Run the script

```python
python3 main.py
```