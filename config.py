# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if it exists

# --- LLM Configuration ---
# Choose LLM_PROVIDER: "ollama" or "chatgpt"
LLM_PROVIDER = "chatgpt"  # Choices: "ollama", "chatgpt"

# Ollama Settings (if LLM_PROVIDER is "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = "llama3:8b"  # e.g., "llama3:8b", "mistral", "llama2"
OLLAMA_EMBEDDING_MODEL = (
    "nomic-embed-text"  # e.g., "nomic-embed-text", "mxbai-embed-large"
)

# OpenAI Settings (if LLM_PROVIDER is "chatgpt")
# Ensure OPENAI_API_KEY is set in your .env file or environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = "gpt-4o-mini"  # e.g., "gpt-3.5-turbo", "gpt-4"
OPENAI_EMBEDDING_MODEL = (
    "text-embedding-3-small"  # e.g., "text-embedding-3-small", "text-embedding-ada-002"
)

# --- RAG Configuration ---
# Choose RAG_TYPE: "standard" or "graph"
RAG_TYPE = "standard"  # Choices: "standard", "graph"

# --- Data Paths ---
PDF_DIRECTORY_PATH = "./docs"  # Directory containing PDF files
QNA_CSV_PATH = "eval_qna/qna_data.csv"  # CSV file with "Question" and "Answer" columns

# --- Vectorstore Configuration ---
VECTORSTORE_COLLECTION_BASE_NAME = "pdf_eval_docs"  # Base name for Chroma collection
CHROMA_PERSIST_DIRECTORY = (
    "./chroma_db_store"  # Directory to persist Chroma DB (set to None to not persist)
)

# --- Text Splitting Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- GraphRAG Specific Configuration (if RAG_TYPE is "graph") ---
GRAPH_EDGES = [("source", "source")]  # Connects chunks from the same PDF.
GRAPH_RETRIEVER_STRATEGY_K = 5
GRAPH_RETRIEVER_STRATEGY_START_K = 3
GRAPH_RETRIEVER_STRATEGY_MAX_DEPTH = 3

# --- Standard RAG Specific Configuration (if RAG_TYPE is "standard") ---
STANDARD_RETRIEVER_K = 5  # Number of documents to retrieve for standard RAG

# --- DeepEval Metrics Configuration ---
# Choose the model provider for DeepEval metrics.
# "ollama": Use the OLLAMA_LLM_MODEL.
# "chatgpt": Use the OPENAI_LLM_MODEL (requires OPENAI_API_KEY).
# "default": DeepEval uses its own default (often OpenAI-based, may require OPENAI_API_KEY).
DEEPEVAL_METRICS_MODEL_PROVIDER = "default"  # Choices: "ollama", "chatgpt", "default"


# --- Helper functions for clarity in the main script ---
def get_vectorstore_collection_name():
    """Generates a unique collection name based on current config."""
    return (
        f"{VECTORSTORE_COLLECTION_BASE_NAME}_{LLM_PROVIDER}_{RAG_TYPE}".lower().replace(
            ":", "_"
        )
    )


def get_llm_model_name_for_logging():
    if LLM_PROVIDER == "ollama":
        return OLLAMA_LLM_MODEL
    elif LLM_PROVIDER == "chatgpt":
        return OPENAI_LLM_MODEL
    return "N/A"


def get_embedding_model_name_for_logging():
    if LLM_PROVIDER == "ollama":
        return OLLAMA_EMBEDDING_MODEL
    elif LLM_PROVIDER == "chatgpt":
        return OPENAI_EMBEDDING_MODEL
    return "N/A"
