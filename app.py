#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import requests

# --- Load Environment Variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)  # Load .env dynamically

api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
cx_code = os.getenv("GOOGLE_CX_CODE")

# --- Debugging: Check if API Keys Loaded ---
if not api_key or not google_api_key or not cx_code:
    st.error("❌ API keys are missing! Please check your .env file.")
    raise ValueError("❌ API keys are missing! Please check your .env file.")

# --- Load FAISS Index (Prevent Error if Index is Missing) ---
faiss_index_path = "faiss_index"
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

if os.path.exists(faiss_index_path):
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    st.warning("⚠️ FAISS index not found. New documents will be stored on first run.")
    vectorstore = FAISS(embeddings)

# --- Function to Retrieve from FAISS ---
def retrieve_faiss_results(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Get top 3 results
    return retriever.invoke(query)

# --- Function to Fetch Google Search Results ---
def fetch_google_results(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={cx_code}"
    
    try:
        response = requests.get(url, timeout=5)  # Set timeout
        response.raise_for_status()  # Raise error for bad responses
        data = response.json()
        
        results = []
        for item in data.get("items", []):
            text = f"{item['title']} - {item['snippet']} ({item['link']})"
            results.append(Document(page_content=text))
        
        return results
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Google Search API Error: {e}")
        return []

# --- Function to Handle Small Talk Queries ---
def handle_small_talk(query):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    return "👋 Hello! How can I assist you today?" if query.lower() in greetings else None

# --- Streamlit UI ---
st.set_page_config(page_title="AI Chatbot with Google Search & FAISS", layout="wide")
st.title("🤖 AI Chatbot with Google Search & FAISS")

# --- Input Box ---
query = st.text_input("🔍 Ask something...", placeholder="e.g., What are the latest AI trends in 2025?")

if st.button("Search & Generate Answer"):
    if query:
        # 🔹 Check for Small Talk
        small_talk_response = handle_small_talk(query)
        if small_talk_response:
            st.success(small_talk_response)
        else:
            with st.spinner("⏳ Fetching results... Please wait."):
                # --- Retrieve from FAISS ---
                retrieved_docs = retrieve_faiss_results(query)

                # --- Fetch Google Search Results & Update FAISS ---
                google_docs = fetch_google_results(query)
                
                # Only add relevant Google results
                relevant_google_docs = [doc for doc in google_docs if len(doc.page_content) > 50]
                if relevant_google_docs:
                    vectorstore.add_documents(relevant_google_docs)
                    vectorstore.save_local(faiss_index_path)

                # --- Display Retrieved Results ---
                if retrieved_docs:
                    st.subheader("🔹 Relevant Documents from FAISS:")
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"**{i+1}.** {doc.page_content}")

                st.subheader("🌐 Top Google Search Results:")
                for i, doc in enumerate(google_docs[:3]):  # Show top 3 results
                    st.write(f"**{i+1}.** {doc.page_content}")

            st.success("✅ Results fetched successfully!")
    else:
        st.warning("⚠️ Please enter a query.")

# --- Footer ---
st.markdown("<br><center>🔹 Made with ❤️ by AI Enthusiast</center>", unsafe_allow_html=True)
