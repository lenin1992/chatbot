#!/usr/bin/env python
# coding: utf-8
"""
app.py
---------------
Streamlit-powered AI Chatbot that:
- Retrieves answers using FAISS vector search.
- Fetches live results from Google Custom Search.
- Handles small talk intelligently.

⚠️ Ensure your .env file contains:
  - OPENAI_API_KEY
  - GOOGLE_API_KEY
  - GOOGLE_CX_CODE
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ✅ Load Environment Variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if not os.path.exists(dotenv_path):
    st.error("❌ .env file is missing! Please create it and add your API keys.")
    raise FileNotFoundError("❌ Missing .env file! Ensure it exists in the project directory.")

load_dotenv(dotenv_path)

# ✅ Retrieve API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cx_code = os.getenv("GOOGLE_CX_CODE")

# ✅ Validate API Keys
if not all([openai_api_key, google_api_key, google_cx_code]):
    st.error("❌ API keys are missing! Please check your .env file.")
    raise ValueError("❌ API keys are missing! Add them in the .env file.")

# ✅ Load FAISS Vector Store
faiss_index_path = "/home/ubuntu/chatbot/faiss_index"  # ✅ Use absolute path
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")

if os.path.exists(faiss_index_path):
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    st.success("✅ FAISS index loaded successfully!")
else:
    st.warning("⚠️ FAISS index not found. A new index will be created on first update.")
    vectorstore = FAISS(embeddings)

# ✅ Function to Retrieve from FAISS
def retrieve_faiss_results(query, top_k=3):
    """
    Retrieves top_k relevant documents from FAISS.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever.invoke(query)

# ✅ Function to Fetch Google Search Results
def fetch_google_results(query, top_n=3):
    """
    Fetches top_n results from Google Custom Search API.
    """
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={google_cx_code}"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            text = f"{item['title']} - {item['snippet']} ({item['link']})"
            results.append(Document(page_content=text))

        return results[:top_n]
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Google Search API Error: {e}")
        return []

# ✅ Function to Handle Small Talk Queries
def handle_small_talk(query):
    """
    Handles simple greetings and common small talk.
    """
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    return "👋 Hello! How can I assist you today?" if query.lower() in greetings else None

# ✅ Streamlit UI
st.set_page_config(page_title="AI Chatbot with Google Search & FAISS", layout="wide")
st.title("🤖 AI Chatbot with Google Search & FAISS")

query = st.text_input("🔍 Ask something...", placeholder="e.g., What are the latest AI trends in 2025?")

if st.button("Search & Generate Answer"):
    if query:
        small_talk_response = handle_small_talk(query)
        if small_talk_response:
            st.success(small_talk_response)
        else:
            with st.spinner("⏳ Fetching results... Please wait."):
                # ✅ Retrieve FAISS Results
                retrieved_docs = retrieve_faiss_results(query)
                
                # ✅ Fetch Google Results
                google_docs = fetch_google_results(query)
                
                # ✅ Update FAISS with Google Results (if relevant)
                if google_docs:
                    vectorstore.add_documents(google_docs)
                    vectorstore.save_local(faiss_index_path)

                # ✅ Display FAISS Results
                if retrieved_docs:
                    st.subheader("🔹 Relevant Documents from FAISS:")
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"**{i+1}.** {doc.page_content}")

                # ✅ Display Google Search Results
                if google_docs:
                    st.subheader("🌐 Top Google Search Results:")
                    for i, doc in enumerate(google_docs):
                        st.write(f"**{i+1}.** {doc.page_content}")
            
            st.success("✅ Results fetched successfully!")
    else:
        st.warning("⚠️ Please enter a query.")

# ✅ Footer
st.markdown("<br><center>🔹 Made with ❤️ by AI Enthusiast</center>", unsafe_allow_html=True)
