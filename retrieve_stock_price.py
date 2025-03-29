#!/usr/bin/env python
# coding: utf-8
"""
retrieval.py
---------------
This script:
- Loads the FAISS vector store.
- Uses OpenAI embeddings for similarity-based retrieval.
- Retrieves the most relevant documents based on a query.

⚠️ Ensure your .env file contains the OPENAI_API_KEY.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS  # ✅ Correct import
from langchain_openai import OpenAIEmbeddings

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Load Environment Variables
env_path = "/home/ubuntu/chatbot/.env"
if not os.path.exists(env_path):
    logging.error("❌ .env file not found! Ensure the correct path.")
    raise FileNotFoundError("❌ .env file missing. Please create it in the correct directory.")

load_dotenv(env_path)

# ✅ Retrieve API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("❌ OPENAI_API_KEY is missing. Please check your .env file.")
    raise ValueError("❌ Missing API key! Add OPENAI_API_KEY in .env.")

# ✅ Load OpenAI Embeddings
try:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    logging.info("✅ OpenAI embeddings initialized.")
except Exception as e:
    logging.error(f"❌ Failed to initialize OpenAI embeddings: {e}")
    raise RuntimeError("❌ OpenAI embeddings initialization failed.")

# ✅ Load FAISS Index
faiss_index_path = "/home/ubuntu/chatbot/faiss_index"  # ✅ Absolute path
try:
    vectorstore = FAISS.load_local(
        folder_path=faiss_index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    logging.info("✅ FAISS vector store loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load FAISS index: {e}")
    raise RuntimeError("❌ FAISS index loading failed. Check file path and integrity.")

# ✅ Convert FAISS into a Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
logging.info("✅ Retriever initialized with FAISS.")

# ✅ Function to Retrieve Documents with Filtering
def retrieve_documents(query_text, top_k=5, similarity_threshold=0.7):
    """
    Retrieves the top_k most relevant documents based on the query_text.
    Filters documents based on a similarity score threshold if available.
    """
    try:
        all_docs = retriever.invoke(query_text)
        if not all_docs:
            logging.warning("⚠️ No relevant documents found.")
            return []

        # ✅ Filter documents based on similarity score if available
        if hasattr(all_docs[0], "metadata") and "score" in all_docs[0].metadata:
            filtered_docs = [doc for doc in all_docs if doc.metadata["score"] >= similarity_threshold]
        else:
            filtered_docs = all_docs  # Fallback if scores are missing

        return filtered_docs[:top_k]
    except Exception as e:
        logging.error(f"❌ Retrieval error: {e}")
        return []

# ✅ Example Query
if __name__ == "__main__":
    query = "Current SBI stock performance and price movement"
    retrieved_docs = retrieve_documents(query)

    # ✅ Display Results
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\n🔹 Retrieved Document {i}:\n{doc.page_content[:500]}")  # Show first 500 chars
    else:
        logging.info("❌ No documents retrieved.")
