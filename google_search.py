#!/usr/bin/env python
# coding: utf-8

import os
import requests
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ✅ Load Environment Variables
env_path = "/home/ubuntu/chatbot/.env"  # Absolute path to .env file
load_dotenv(env_path)

# ✅ Retrieve API Keys from .env
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("API_KEY")  # ✅ Renamed to match .env
cx_code = os.getenv("CX_CODE")  # ✅ Renamed to match .env

# ✅ Ensure All API Keys are Loaded
if not openai_api_key or not google_api_key or not cx_code:
    raise ValueError("❌ Required API keys are missing! Please check your .env file.")

# ✅ Function to Fetch Google Search Results
def fetch_google_results(query):
    """Fetch top 10 search results from Google Custom Search API."""
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={cx_code}"
    response = requests.get(url)
    data = response.json()
    
    results = []
    for item in data.get("items", []):
        text = f"{item['title']} - {item['snippet']} ({item['link']})"
        results.append(Document(page_content=text))
    
    return results

# ✅ Function to Update FAISS Index with Google Search Data
def update_faiss_with_google(query, faiss_index_path="/home/ubuntu/chatbot/faiss_index"):
    """Fetch Google results and store them in FAISS."""
    
    # Load existing FAISS index (or create a new one)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Fetch Google Search results
    google_docs = fetch_google_results(query)
    
    # Split into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(google_docs)
    
    # Add to FAISS
    vectorstore.add_documents(chunks)
    
    # Save updated FAISS index
    vectorstore.save_local(faiss_index_path)
    print(f"🔹 Google search data added to FAISS for query: {query}")

# ✅ Function to Retrieve Relevant Documents from FAISS
def retrieve_from_faiss(query, faiss_index_path="/home/ubuntu/chatbot/faiss_index"):
    """Retrieve relevant documents from FAISS."""
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)

    # Print Retrieved Documents
    for i, doc in enumerate(retrieved_docs):
        print(f"\n🔹 Retrieved Document {i+1}:\n{doc.page_content[:500]}")

# ✅ Example Usage:
query = "latest AI trends 2025"
update_faiss_with_google(query)  # Update FAISS with new search results
retrieve_from_faiss(query)       # Retrieve data from FAISS
