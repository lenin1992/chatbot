#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv  # ✅ Import load_dotenv
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ✅ Load Environment Variables
env_path = "/home/ubuntu/chatbot/.env"  # Absolute path to .env file
load_dotenv(env_path)

# Ensure OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Load OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Load FAISS index with safe deserialization
faiss_index_path = "faiss_index"  # Adjust to your actual FAISS path
vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# Convert FAISS into a retriever with filtering
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Function to retrieve documents with better filters
def retrieve_documents(query_text, top_k=5, similarity_threshold=0.7):
    all_docs = retriever.invoke(query_text)
    
    # Filter based on a similarity score threshold (if scores are available)
    if hasattr(all_docs[0], "metadata") and "score" in all_docs[0].metadata:
        filtered_docs = [doc for doc in all_docs if doc.metadata["score"] >= similarity_threshold]
    else:
        filtered_docs = all_docs  # Fallback if scores are missing
    
    return filtered_docs[:top_k]

# Enhanced query example (with synonyms)
query = "Current SBI stock performance and price movement"
retrieved_docs = retrieve_documents(query)

# Display results
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n🔹 Retrieved Document {i}:\n{doc.page_content[:500]}")  # Show first 500 chars
