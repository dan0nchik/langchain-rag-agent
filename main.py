import os
import pandas as pd
import shutil
import wonderwords
import gradio as gr  # Added Gradio
import logging

# Langchain components
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# LLM and Embeddings
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# GraphRAG components
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.transformers import ShreddingTransformer
from langchain_graph_retriever.adapters.chroma import ChromaAdapter
from graph_retriever.strategies import Eager

# Langchain Core
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)  # Added for potential future use with history

# Import configurations
import config  # Make sure this file exists and is correctly configured

# --- Global variable for the RAG chain ---
# This will be initialized by setup_rag_pipeline()
rag_chain = None
# ---

r = wonderwords.RandomWord()
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)
# Ensure logger has handlers only if not already configured (e.g. by Gradio)
if not logger.hasHandlers():
    logging.basicConfig(
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",  # Added format
    )


def setup_rag_pipeline():
    """
    Sets up the entire RAG pipeline based on config settings.
    Returns the initialized RAG chain.
    """
    global rag_chain  # To assign to the global variable

    logging.info("--- Configuration Summary (for RAG pipeline) ---")
    logging.info(f"LLM Provider: {config.LLM_PROVIDER}")
    logging.info(f"LLM Model: {config.get_llm_model_name_for_logging()}")
    logging.info(f"Embedding Model: {config.get_embedding_model_name_for_logging()}")
    logging.info(f"RAG Type: {config.RAG_TYPE}")
    logging.info(f"PDF Directory: {config.PDF_DIRECTORY_PATH}")
    logging.info(f"Vector Store Collection: {config.get_vectorstore_collection_name()}")
    if config.CHROMA_PERSIST_DIRECTORY:
        logging.info(f"Chroma Persist Directory: {config.CHROMA_PERSIST_DIRECTORY}")
    logging.info("-----------------------------")

    # Validate PDF directory
    if not os.path.exists(config.PDF_DIRECTORY_PATH) or not os.listdir(
        config.PDF_DIRECTORY_PATH
    ):
        msg = f"Error: PDF directory '{config.PDF_DIRECTORY_PATH}' is empty or does not exist."
        logging.error(msg)
        raise FileNotFoundError(f"{msg} Please create it and add your PDF files.")

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
        msg = f"Error: No documents loaded from '{config.PDF_DIRECTORY_PATH}'. Check PDF files."
        logging.error(msg)
        raise ValueError(msg)
    logging.info(f"Loaded {len(documents)} PDF documents (pages/files).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    doc_chunks = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(doc_chunks)} document chunks.")

    for chunk in doc_chunks:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown_source"
        else:
            chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])

    # --- 3. Setup Retriever based on RAG_TYPE ---
    retriever = None
    vectorstore_collection_name = config.get_vectorstore_collection_name()

    if config.CHROMA_PERSIST_DIRECTORY and os.path.exists(
        os.path.join(config.CHROMA_PERSIST_DIRECTORY, vectorstore_collection_name)
    ):
        logging.warning(
            f"Warning: Existing Chroma collection '{vectorstore_collection_name}' found in '{config.CHROMA_PERSIST_DIRECTORY}'. It might be reused or overwritten."
        )

    vectorstore_params = {
        "embedding": embedding_model,
        "collection_name": vectorstore_collection_name,
    }
    if config.CHROMA_PERSIST_DIRECTORY:
        vectorstore_params["persist_directory"] = config.CHROMA_PERSIST_DIRECTORY

    vectorstore = None  # Initialize vectorstore

    if config.RAG_TYPE == "standard":
        logging.info("Setting up Standard RAG...")
        vectorstore = Chroma.from_documents(documents=doc_chunks, **vectorstore_params)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.STANDARD_RETRIEVER_K}
        )
        logging.info(
            f"Standard RAG using Chroma vector store with k={config.STANDARD_RETRIEVER_K}."
        )

    elif config.RAG_TYPE == "graph":
        logging.info("Setting up GraphRAG...")
        logging.info("Shredding documents for graph structure...")
        shredded_docs = list(ShreddingTransformer().transform_documents(doc_chunks))
        logging.info(f"Shredded into {len(shredded_docs)} nodes/chunks for GraphRAG.")

        vectorstore = Chroma.from_documents(
            documents=shredded_docs, **vectorstore_params
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

    if config.CHROMA_PERSIST_DIRECTORY and vectorstore:
        logging.info(
            f"Vector store persisting to: {config.CHROMA_PERSIST_DIRECTORY}/{vectorstore_collection_name}"
        )
        vectorstore.persist()

    # --- 4. Setup LLM Chain ---
    prompt_template_str = """Answer the question based only on the context provided.
If the context does not contain the answer, state "I don't have enough information from the provided context to answer."
Be concise and precise. Cite the source document if possible, like (SOURCE: filename.pdf).

Context:
{context}

Question:
{question}"""
    prompt_template = ChatPromptTemplate.from_template(prompt_template_str)

    def format_docs_for_context(docs):
        if not docs:
            return "No context retrieved."
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}"
            for doc in docs
        )

    # Assign to the global rag_chain
    rag_chain = (
        {
            "context": retriever | format_docs_for_context,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    logging.info("RAG chain setup complete.")
    return rag_chain  # Return it for explicitness, though global is set


def run_deepeval_evaluation():
    """
    Runs the DeepEval evaluation part of the original script.
    This function can be called separately if evaluation is needed.
    Assumes `rag_chain` and `retriever` are already initialized by `setup_rag_pipeline`.
    """
    if rag_chain is None:  # Retriever is part of rag_chain setup
        logging.error("RAG chain not initialized. Call setup_rag_pipeline() first.")
        return

    # --- 5. Load Q&A Data from CSV ---
    logging.info(f"Loading Q&A data from {config.QNA_CSV_PATH} for evaluation...")
    if not os.path.exists(config.QNA_CSV_PATH):
        logging.error(
            f"Error: Q&A CSV file '{config.QNA_CSV_PATH}' not found for evaluation."
        )
        return

    try:
        qna_df = pd.read_csv(config.QNA_CSV_PATH)
        if not all(col in qna_df.columns for col in ["Question", "Answer"]):
            logging.error(
                "Error: CSV must contain 'Question' and 'Answer' columns for evaluation."
            )
            return
    except Exception as e:
        logging.error(f"Error reading CSV file '{config.QNA_CSV_PATH}': {e}")
        return
    logging.info(f"Loaded {len(qna_df)} Q&A pairs for evaluation.")

    # --- 6. Prepare DeepEval Test Cases ---
    # Need the retriever from the chain's components to get retrieved_docs
    # This is a bit indirect; ideally, retriever would be passed or accessible.
    # For now, let's assume the `retriever` variable from `setup_rag_pipeline` is in scope
    # or we re-access it. If `setup_rag_pipeline` is run in a different scope,
    # this retriever would need to be passed to `run_deepeval_evaluation`.
    # For simplicity here, we'll assume it was set up in the same scope.
    # This is a slight simplification; in a more robust app, you'd pass `retriever` explicitly.
    # A quick way to get the retriever if it's part of the global rag_chain:
    # `retriever = rag_chain.first # This gets the first element of the sequence, which is the parallel dict
    # retriever_runnable = retriever.get("context").first # Gets the retriever component
    # This depends on LCEL structure, which can change. Best to pass it.
    # For this example, we'll rely on `retriever` being available from `setup_rag_pipeline` if run in same script.
    # Let's refine `setup_rag_pipeline` to return retriever as well for this.

    # Re-accessing retriever. This is not ideal but works if setup_rag_pipeline was just run.
    # A better way would be to have setup_rag_pipeline return retriever too.
    # For this example, let's assume `retriever` is somehow available (e.g., make it global or return it)
    # We'll modify `setup_rag_pipeline` to make `retriever` accessible (e.g., return it)
    # And modify the call to `setup_rag_pipeline` to get it.
    # For now, to keep it simple, we just log a warning if it's not found.
    current_retriever = None
    try:
        # Attempt to access the retriever from the chain (this is fragile)
        if (
            rag_chain
            and hasattr(rag_chain, "first")
            and isinstance(rag_chain.first, dict)
            and "context" in rag_chain.first
        ):
            context_runnable = rag_chain.first["context"]
            if hasattr(context_runnable, "first"):  # Check if it's a Sequence
                current_retriever = context_runnable.first
    except Exception as e:
        logging.warning(
            f"Could not automatically extract retriever from rag_chain for DeepEval: {e}"
        )

    if current_retriever is None:
        logging.error(
            "Retriever not available for DeepEval. Evaluation cannot proceed with retrieval_context."
        )
        # You could still run some metrics if you mock retrieval_context or skip context-based ones.
        # For now, we'll just not create retrieval_context if retriever is missing.

    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        AnswerRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import GPTModel, OllamaModel
    from deepeval.test_case import LLMTestCase
    from deepeval import evaluate

    test_cases = []
    logging.info("Preparing DeepEval test cases...")
    for index, row in qna_df.iterrows():
        query = str(row["Question"])
        expected_answer = str(row["Answer"])
        logging.info(f"\nProcessing Q{index+1} for eval: {query[:60]}...")
        try:
            retrieval_context_list = []
            if current_retriever:
                retrieved_docs = current_retriever.invoke(
                    query, config=RunnableConfig(run_name="Retriever_for_eval")
                )
                retrieval_context_list = [doc.page_content for doc in retrieved_docs]
            else:
                retrieval_context_list = ["Retriever not available for this test case."]

            actual_output = rag_chain.invoke(
                query, config=RunnableConfig(run_name="FullRAGChain_for_eval")
            )

            test_case = LLMTestCase(
                input=query,
                actual_output=(
                    actual_output if actual_output else "Error: Empty LLM output"
                ),
                expected_output=expected_answer,
                retrieval_context=(
                    retrieval_context_list
                    if retrieval_context_list
                    else ["No documents retrieved / retriever unavailable."]
                ),
            )
            test_cases.append(test_case)
            logging.info(f"  Input: {query}")
            logging.info(f"  Expected Answer: {expected_answer[:100]}...")
            logging.info(
                f"  Actual LLM Output: {actual_output[:100] if actual_output else 'None'}..."
            )
            logging.info(
                f"  Retrieved {len(retrieval_context_list)} contexts for evaluation."
            )

        except Exception as e:
            logging.error(f"  Error processing question '{query}' for eval: {e}")
            test_cases.append(
                LLMTestCase(
                    input=query,
                    actual_output=f"Error during RAG generation for eval: {e}",
                    expected_output=expected_answer,
                    retrieval_context=[],
                )
            )

    if not test_cases:
        logging.info("No test cases were generated for DeepEval. Exiting evaluation.")
        return

    # --- 7. Define and Run DeepEval Evaluation ---
    logging.info("\n--- Starting DeepEval Evaluation ---")
    eval_llm_for_metric = None
    deepeval_model_name_for_metric_logging = "N/A"

    if config.DEEPEVAL_METRICS_MODEL_PROVIDER == "ollama":
        if not config.OLLAMA_LLM_MODEL:
            raise ValueError(
                "OLLAMA_LLM_MODEL must be set for DEEPEVAL_METRICS_MODEL_PROVIDER='ollama'"
            )
        eval_llm_for_metric = OllamaModel(
            model=config.OLLAMA_LLM_MODEL, base_url=config.OLLAMA_BASE_URL
        )
        deepeval_model_name_for_metric_logging = f"Ollama: {config.OLLAMA_LLM_MODEL}"
    elif config.DEEPEVAL_METRICS_MODEL_PROVIDER == "chatgpt":
        if not config.OPENAI_LLM_MODEL:
            raise ValueError(
                "OPENAI_LLM_MODEL must be set for DEEPEVAL_METRICS_MODEL_PROVIDER='chatgpt'"
            )
        eval_llm_for_metric = GPTModel(
            model=config.DEEPEVAL_METRICS_GPT_MODEL
        )  # Uses OPENAI_API_KEY from env
        deepeval_model_name_for_metric_logging = f"OpenAI: {config.OPENAI_LLM_MODEL}"
    elif config.DEEPEVAL_METRICS_MODEL_PROVIDER == "default":
        deepeval_model_name_for_metric_logging = "DeepEval Default"
        if not config.OPENAI_API_KEY:
            logging.warning(
                "DEEPEVAL_METRICS_MODEL_PROVIDER is 'default'. If it uses OpenAI, API key should be set."
            )
    else:
        raise ValueError(
            f"Unsupported DEEPEVAL_METRICS_MODEL_PROVIDER: {config.DEEPEVAL_METRICS_MODEL_PROVIDER}"
        )

    logging.info(
        f"Initializing DeepEval metrics using model: {deepeval_model_name_for_metric_logging}"
    )

    metrics_to_run = [
        ContextualPrecisionMetric(model=eval_llm_for_metric, include_reason=True),
        ContextualRecallMetric(model=eval_llm_for_metric, include_reason=True),
        ContextualRelevancyMetric(model=eval_llm_for_metric, include_reason=True),
        AnswerRelevancyMetric(model=eval_llm_for_metric, include_reason=True),
        FaithfulnessMetric(model=eval_llm_for_metric, include_reason=True),
    ]
    evaluation_results = evaluate(test_cases=test_cases, metrics=metrics_to_run)
    logging.info(
        f"DeepEval Results URL (if enabled and logged in): {evaluation_results.confident_link}"
    )
    logging.info("\n✅ Evaluation Complete.")

    # Optional: Clean up Chroma vector store (if you want to do it after evaluation)
    # This is usually done outside the evaluation function, perhaps at script end.
    # client = Chroma(
    #     persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    #     collection_name=config.get_vectorstore_collection_name(),
    # )
    # try:
    #     logging.info(f"\nAttempting to delete Chroma collection: {config.get_vectorstore_collection_name()}")
    #     client.delete_collection()
    #     logging.info(f"Chroma collection '{config.get_vectorstore_collection_name()}' deleted.")
    # except Exception as e:
    #     logging.info(f"Could not delete collection '{config.get_vectorstore_collection_name()}': {e}")


# --- Gradio UI Integration ---
def predict_with_rag(message, history):
    """
    Prediction function for Gradio ChatInterface.
    Uses the globally initialized `rag_chain`.
    The `history` parameter is provided by Gradio but not directly used by this RAG chain,
    as the RAG chain is stateless per query. Gradio handles displaying history.
    """
    if rag_chain is None:
        logging.error("RAG chain is not initialized. Cannot predict.")
        return "Error: RAG system not ready. Please check logs."

    logging.info(f"Received user message: {message}")
    # If you wanted to make the RAG chain conversational using the history:
    # 1. Modify `prompt_template_str` to include a placeholder for chat_history.
    # 2. Convert `history` (list of dicts) to Langchain `HumanMessage`/`AIMessage` list.
    # 3. Update `rag_chain` to accept `chat_history` and pass it to the prompt.
    # For now, it's stateless for simplicity, relying on retriever for context each time.

    response = rag_chain.invoke(message)
    logging.info(f"Generated RAG response: {response}")
    return response


if __name__ == "__main__":
    # --- 0. Initial Setup & Configuration Validation ---
    # Moved config logging and validation into setup_rag_pipeline for RAG specific parts
    # Basic logging setup happens at the top level.

    # Setup the RAG pipeline (this will initialize the global `rag_chain`)
    try:
        setup_rag_pipeline()
        logging.info("RAG Pipeline setup successful.")
    except Exception as e:
        logging.error(f"Failed to setup RAG pipeline: {e}", exc_info=True)
        print(
            f"Critical Error: Failed to setup RAG pipeline. Check logs. Exiting. Error: {e}"
        )
        exit(1)  # Exit if pipeline setup fails

    if config.RUN_DEEPEVAL_EVALUATION:
        try:
            run_deepeval_evaluation()
        except Exception as e:
            logging.error(f"Failed to run DeepEval evaluation: {e}", exc_info=True)
            print(
                f"Error: Failed to run DeepEval evaluation. Check logs. Exiting. Error: {e}"
            )
            exit(1)
    # Launch Gradio UI
    logging.info(
        f"Launching Gradio Chat Interface with {config.RAG_TYPE.upper()} RAG..."
    )
    print(f"Launching Gradio Chat Interface with {config.RAG_TYPE.upper()} RAG...")
    print(
        f"LLM: {config.get_llm_model_name_for_logging()}, Embeddings: {config.get_embedding_model_name_for_logging()}"
    )
    print(f"PDFs from: {config.PDF_DIRECTORY_PATH}")

    # Ensure OPENAI_API_KEY is available for Gradio's default OpenAI model if used by themes/etc.
    # Though our RAG chain uses its own LLM.
    if not os.getenv("OPENAI_API_KEY") and config.LLM_PROVIDER == "chatgpt":
        os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
    elif not os.getenv("OPENAI_API_KEY") and config.LLM_PROVIDER != "chatgpt":
        # Set a dummy key if OpenAI is not the provider to prevent Gradio warnings/errors
        # if it tries to use some OpenAI feature by default (e.g. for telemetry or certain themes)
        os.environ["OPENAI_API_KEY"] = "sk-dummykeyforgradionoopenaiused"
        logging.info(
            "Set dummy OPENAI_API_KEY for Gradio as OpenAI is not the primary LLM provider."
        )

    demo = gr.ChatInterface(
        fn=predict_with_rag,
        title=f"Document Q&A ({config.RAG_TYPE.capitalize()} RAG)",
        description=(
            f"Ask questions about the documents loaded from '{config.PDF_DIRECTORY_PATH}'.\n"
            f"Using LLM: {config.get_llm_model_name_for_logging()} via {config.LLM_PROVIDER}.\n"
            f"Embeddings: {config.get_embedding_model_name_for_logging()}."
        ),
        type="messages",  # Passes history as a list of dicts: [{'role': 'user', 'content': '...'}, ...]
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(
            placeholder="Ask your question here...", container=False, scale=7
        ),
        submit_btn="Ask",
        examples=(
            [
                "What is the main topic of the documents?",
                "Summarize the key findings regarding project X.",
                # Add more relevant examples based on your documents
            ]
            if config.PDF_DIRECTORY_PATH
            else None
        ),  # Only show examples if PDF path is configured
    )

    # To make it accessible on the network, use share=True (be cautious with this)
    # and server_name="0.0.0.0"
    demo.launch()  # server_name="0.0.0.0", server_port=7860, share=False

    logging.info("Gradio UI has been launched.")
    # Note: The script will block here until Gradio is closed.
    # Code after demo.launch() will only run after the UI server is stopped.

    # Optional: Clean up Chroma vector store after UI is closed
    # This part will run only after you stop the Gradio server (e.g., Ctrl+C in terminal)
    # You might want to make this conditional or part of a separate cleanup script.
    if config.CHROMA_PERSIST_DIRECTORY:  # Set to True to enable cleanup on exit
        logging.info("Gradio UI closed. Attempting to clean up Chroma vector store...")
        vectorstore_collection_name = config.get_vectorstore_collection_name()
        try:
            # Re-initialize client to connect to the persisted store
            client = Chroma(
                persist_directory=config.CHROMA_PERSIST_DIRECTORY,
                # embedding_function is not strictly needed for delete_collection but good practice if recreating
                # embedding_function=embedding_model # embedding_model might not be in scope here
                # if setup_rag_pipeline was self-contained.
                # Best to just pass persist_dir and collection_name for deletion.
            )
            logging.info(
                f"Attempting to delete Chroma collection: {vectorstore_collection_name}"
            )
            client.delete_collection()
            logging.info(f"Chroma collection '{vectorstore_collection_name}' deleted.")
        except Exception as e:
            logging.error(
                f"Could not delete collection '{vectorstore_collection_name}' on exit: {e}",
                exc_info=True,
            )
