import os
import pandas as pd
import shutil  # For cleaning up Chroma persist directory
import wonderwords

# Langchain components
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # Or from langchain_text_splitters
from langchain_community.vectorstores import Chroma

# LLM and Embeddings
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# GraphRAG components
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.transformers import ShreddingTransformer
from langchain_graph_retriever.adapters.chroma import ChromaAdapter
from graph_retriever.strategies import (
    Eager,
)

# Langchain Core
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig

# DeepEval
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# Import configurations
import config

import logging

r = wonderwords.RandomWord()
os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"logs/{'-'.join(r.random_words(3))}.log",
    encoding="utf-8",
    level=logging.INFO,
)

# --- 0. Initial Setup & Configuration Validation ---
logging.info("--- Configuration Summary ---")
logging.info(f"LLM Provider: {config.LLM_PROVIDER}")
logging.info(f"LLM Model: {config.get_llm_model_name_for_logging()}")
logging.info(f"Embedding Model: {config.get_embedding_model_name_for_logging()}")
logging.info(f"RAG Type: {config.RAG_TYPE}")
logging.info(f"PDF Directory: {config.PDF_DIRECTORY_PATH}")
logging.info(f"QnA CSV Path: {config.QNA_CSV_PATH}")
logging.info(f"Vector Store Collection: {config.get_vectorstore_collection_name()}")
if config.CHROMA_PERSIST_DIRECTORY:
    logging.info(f"Chroma Persist Directory: {config.CHROMA_PERSIST_DIRECTORY}")
logging.info("-----------------------------")


# Validate PDF directory
if not os.path.exists(config.PDF_DIRECTORY_PATH) or not os.listdir(
    config.PDF_DIRECTORY_PATH
):
    logging.info(
        f"Error: PDF directory '{config.PDF_DIRECTORY_PATH}' is empty or does not exist."
    )
    logging.info("Please create it and add your PDF files.")
    exit()

# Validate QnA CSV
if not os.path.exists(config.QNA_CSV_PATH):
    logging.info(f"Error: Q&A CSV file '{config.QNA_CSV_PATH}' not found.")
    exit()

# --- 1. Initialize LLM and Embedding Model ---
embedding_model = None
llm = None

if config.LLM_PROVIDER == "ollama":
    embedding_model = OllamaEmbeddings(
        model=config.OLLAMA_EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    llm = ChatOllama(model=config.OLLAMA_LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
elif config.LLM_PROVIDER == "chatgpt":
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in .env or environment variables when using 'chatgpt' provider."
        )
    embedding_model = OpenAIEmbeddings(
        api_key=config.OPENAI_API_KEY, model=config.OPENAI_EMBEDDING_MODEL
    )
    llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model=config.OPENAI_LLM_MODEL)
else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")

logging.info(
    f"Initialized LLM: {config.get_llm_model_name_for_logging()} from {config.LLM_PROVIDER}"
)
logging.info(
    f"Initialized Embedding Model: {config.get_embedding_model_name_for_logging()}"
)

# --- 2. Load and Process Documents ---
logging.info(f"Loading PDFs from {config.PDF_DIRECTORY_PATH}...")
loader = PyPDFDirectoryLoader(config.PDF_DIRECTORY_PATH)
documents = loader.load()

if not documents:
    logging.info(
        f"Error: No documents loaded from '{config.PDF_DIRECTORY_PATH}'. Check PDF files."
    )
    exit()
logging.info(f"Loaded {len(documents)} PDF documents (pages/files).")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
)
doc_chunks = text_splitter.split_documents(documents)
logging.info(f"Split into {len(doc_chunks)} document chunks.")

# Ensure 'source' metadata is present for citation and graph edges
for chunk in doc_chunks:
    if "source" not in chunk.metadata:
        chunk.metadata["source"] = "unknown_source"
    else:  # Make source more readable (e.g., just filename)
        chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])


# --- 3. Setup Retriever based on RAG_TYPE ---
retriever = None
vectorstore_collection_name = config.get_vectorstore_collection_name()

# Optionally, clean up previous DB for this specific collection if persisting
if config.CHROMA_PERSIST_DIRECTORY and os.path.exists(
    os.path.join(config.CHROMA_PERSIST_DIRECTORY, vectorstore_collection_name)
):
    logging.info(
        f"Warning: Existing Chroma collection '{vectorstore_collection_name}' found in '{config.CHROMA_PERSIST_DIRECTORY}'. It might be reused or overwritten depending on Chroma's behavior."
    )
    # For a full clean, you might need to delete the specific collection directory within chroma_db_store or use client.delete_collection
    # For simplicity, we'll let Chroma handle it or you can manually clear `config.CHROMA_PERSIST_DIRECTORY`

vectorstore_params = {
    "embedding": embedding_model,
    "collection_name": vectorstore_collection_name,
}
if config.CHROMA_PERSIST_DIRECTORY:
    vectorstore_params["persist_directory"] = config.CHROMA_PERSIST_DIRECTORY


if config.RAG_TYPE == "standard":
    logging.info("Setting up Standard RAG...")
    vectorstore = Chroma.from_documents(
        documents=doc_chunks, **vectorstore_params  # Use directly split chunks
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.STANDARD_RETRIEVER_K}
    )
    logging.info(
        f"Standard RAG using Chroma vector store with k={config.STANDARD_RETRIEVER_K}."
    )

elif config.RAG_TYPE == "graph":
    logging.info("Setting up GraphRAG...")
    logging.info("Shredding documents for graph structure...")
    # ShreddingTransformer expects list of Document objects
    shredded_docs = list(
        ShreddingTransformer().transform_documents(doc_chunks)
    )  # ensure_metadata_DNE_label for robustness
    logging.info(f"Shredded into {len(shredded_docs)} nodes/chunks for GraphRAG.")

    vectorstore = Chroma.from_documents(
        documents=shredded_docs, **vectorstore_params  # Use shredded docs for GraphRAG
    )
    chroma_adapter = ChromaAdapter(vector_store=vectorstore)

    retriever = GraphRetriever(
        store=chroma_adapter,
        edges=config.GRAPH_EDGES,
        strategy=Eager(
            k=config.GRAPH_RETRIEVER_STRATEGY_K,
            start_k=config.GRAPH_RETRIEVER_STRATEGY_START_K,
            max_depth=config.GRAPH_RETRIEVER_STRATEGY_MAX_DEPTH,
        ),
    )
    logging.info("GraphRAG retriever setup complete.")
else:
    raise ValueError(f"Unsupported RAG_TYPE: {config.RAG_TYPE}")

if config.CHROMA_PERSIST_DIRECTORY:
    logging.info(
        f"Vector store persisting to: {config.CHROMA_PERSIST_DIRECTORY}/{vectorstore_collection_name}"
    )
    vectorstore.persist()  # Ensure persistence if directory is set

# --- 4. Setup LLM Chain ---
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.
If the context does not contain the answer, state "I don't have enough information from the provided context to answer."
Be concise and precise. Cite the source document if possible, like (SOURCE: filename.pdf).

Context:
{context}

Question:
{question}"""
)


def format_docs_for_context(docs):
    if not docs:
        return "No context retrieved."
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}"
        for doc in docs
    )


rag_chain = (
    {"context": retriever | format_docs_for_context, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# --- 5. Load Q&A Data from CSV ---
logging.info(f"Loading Q&A data from {config.QNA_CSV_PATH}...")
try:
    qna_df = pd.read_csv(config.QNA_CSV_PATH)
    if not all(col in qna_df.columns for col in ["Question", "Answer"]):
        logging.info("Error: CSV must contain 'Question' and 'Answer' columns.")
        exit()
except Exception as e:
    logging.info(f"Error reading CSV file '{config.QNA_CSV_PATH}': {e}")
    exit()
logging.info(f"Loaded {len(qna_df)} Q&A pairs.")

# --- 6. Prepare DeepEval Test Cases ---
test_cases = []
logging.info("Preparing DeepEval test cases...")
for index, row in qna_df.iterrows():
    query = str(row["Question"])  # Ensure query is string
    expected_answer = str(row["Answer"])  # Ensure answer is string

    logging.info(f"\nProcessing Q{index+1}: {query[:60]}...")
    try:
        # Get retrieved documents separately for the 'retrieval_context'
        retrieved_docs = retriever.invoke(
            query, config=RunnableConfig(run_name="Retriever")
        )
        retrieval_context_list = [doc.page_content for doc in retrieved_docs]

        # Get the actual output from the RAG chain
        # Note: The chain invokes the retriever again. For strict context evaluation,
        # you might want to pass the formatted_context directly to a sub-chain.
        # However, for end-to-end eval, invoking the full chain is standard.
        actual_output = rag_chain.invoke(
            query, config=RunnableConfig(run_name="FullRAGChain")
        )

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output if actual_output else "Error: Empty LLM output",
            expected_output=expected_answer,
            retrieval_context=(
                retrieval_context_list
                if retrieval_context_list
                else ["No documents retrieved."]
            ),
        )
        test_cases.append(test_case)
        logging.info(f"  Input: {query}")
        # logging.info(f"  Retrieved Docs Meta: {[doc.metadata for doc in retrieved_docs]}")
        logging.info(f"  Expected Answer: {expected_answer[:100]}...")
        logging.info(
            f"  Actual LLM Output: {actual_output[:100] if actual_output else 'None'}..."
        )
        logging.info(
            f"  Retrieved {len(retrieval_context_list)} contexts for evaluation."
        )

    except Exception as e:
        logging.info(f"  Error processing question '{query}': {e}")
        # Create a test case with error info
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=f"Error during RAG generation: {e}",
                expected_output=expected_answer,
                retrieval_context=[],
            )
        )

if not test_cases:
    logging.info("No test cases were generated. Exiting evaluation.")
    exit()

# --- 7. Define and Run DeepEval Evaluation ---
logging.info("\n--- Starting DeepEval Evaluation ---")

eval_llm_for_metric = None
deepeval_model_name_for_metric_logging = "N/A"

if config.DEEPEVAL_METRICS_MODEL_PROVIDER == "ollama":
    if not config.OLLAMA_LLM_MODEL:
        raise ValueError(
            "OLLAMA_LLM_MODEL must be set in config for DEEPEVAL_METRICS_MODEL_PROVIDER='ollama'"
        )
    eval_llm_for_metric = ChatOllama(
        model=config.OLLAMA_LLM_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    deepeval_model_name_for_metric_logging = f"Ollama: {config.OLLAMA_LLM_MODEL}"
elif config.DEEPEVAL_METRICS_MODEL_PROVIDER == "chatgpt":
    if not config.OPENAI_LLM_MODEL:
        raise ValueError(
            "OPENAI_LLM_MODEL must be set in config for DEEPEVAL_METRICS_MODEL_PROVIDER='chatgpt'"
        )
    if not config.OPENAI_API_KEY:
        logging.info(
            "Warning: OPENAI_API_KEY is not set. DeepEval metrics configured for 'chatgpt' may fail."
        )
    eval_llm_for_metric = ChatOpenAI(
        model=config.OPENAI_LLM_MODEL, api_key=config.OPENAI_API_KEY
    )
    deepeval_model_name_for_metric_logging = f"OpenAI: {config.OPENAI_LLM_MODEL}"
elif config.DEEPEVAL_METRICS_MODEL_PROVIDER == "default":
    deepeval_model_name_for_metric_logging = "DeepEval Default (likely OpenAI)"
    if not config.OPENAI_API_KEY:  # Check as DeepEval default often uses OpenAI
        logging.info(
            "Warning: DEEPEVAL_METRICS_MODEL_PROVIDER is 'default'. If DeepEval defaults to an OpenAI model, OPENAI_API_KEY should be set."
        )
else:
    raise ValueError(
        f"Unsupported DEEPEVAL_METRICS_MODEL_PROVIDER: {config.DEEPEVAL_METRICS_MODEL_PROVIDER}"
    )

logging.info(
    f"Initializing DeepEval metrics using model: {deepeval_model_name_for_metric_logging}"
)

# Pass the LLM instance (eval_llm_for_metric) to metrics if configured, else DeepEval uses its default.
metrics_to_run = [
    ContextualPrecisionMetric(model=eval_llm_for_metric, include_reason=True),
    ContextualRecallMetric(model=eval_llm_for_metric, include_reason=True),
    ContextualRelevancyMetric(
        model=eval_llm_for_metric, include_reason=True
    ),  # Relevance of context to query
    AnswerRelevancyMetric(
        model=eval_llm_for_metric, include_reason=True
    ),  # Relevance of answer to query
    FaithfulnessMetric(
        model=eval_llm_for_metric, include_reason=True
    ),  # Answer grounded in context
]

# DeepEval evaluation can be lengthy
evaluation_results = evaluate(test_cases=test_cases, metrics=metrics_to_run)

evaluation_results.confident_link
# For more detailed per-test_case results, you can inspect `evaluation_results`
# structure or iterate through `test_cases` which get updated by `evaluate`.
# For example, each `LLMTestCase` object in `test_cases` list will have a `metrics_metadata` attribute after evaluation.
# logging.info("\n--- Detailed Per-Test Case Breakdown (from test_cases) ---")
# for i, tc in enumerate(test_cases):
#     logging.info(f"\nTest Case #{i+1}: {tc.input[:50]}...")
#     if hasattr(tc, 'metrics_metadata') and tc.metrics_metadata:
#         for meta in tc.metrics_metadata:
#             logging.info(f"  - Metric: {meta.metric}, Score: {meta.score:.4f}, Reason: {meta.reason}")
#     else:
#         logging.info("  No metric metadata found on test case (or evaluation failed for this TC).")


logging.info("\n✅ Evaluation Complete.")

# --- Optional: Clean up Chroma vector store ---
# If you want to delete the specific collection used in this run:
client = Chroma(
    persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    collection_name=vectorstore_collection_name,
)
try:
    logging.info(
        f"\nAttempting to delete Chroma collection: {vectorstore_collection_name}"
    )
    client.delete_collection()  # Chroma's API for this might be client.delete_collection(name=...)
    logging.info(f"Chroma collection '{vectorstore_collection_name}' deleted.")
except Exception as e:
    logging.info(f"Could not delete collection '{vectorstore_collection_name}': {e}")
# If you want to wipe the entire persist directory (USE WITH CAUTION):
# if config.CHROMA_PERSIST_DIRECTORY and os.path.exists(config.CHROMA_PERSIST_DIRECTORY):
#     logging.info(f"Wiping Chroma persist directory: {config.CHROMA_PERSIST_DIRECTORY}")
#     shutil.rmtree(config.CHROMA_PERSIST_DIRECTORY)
#     logging.info("Chroma persist directory wiped.")
