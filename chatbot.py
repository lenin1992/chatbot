#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI  # ✅ Updated import for compatibility

# --- Load Environment Variables ---
env_path = "/home/ubuntu/chatbot/.env"  # ✅ Set absolute path
if not load_dotenv(env_path):
    raise ValueError("❌ Failed to load .env file. Check the file path and format.")

# --- Retrieve API Keys from .env ---
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("API_KEY")
cx_code = os.getenv("CX_CODE")

# --- Ensure All API Keys are Loaded ---
if not all([openai_api_key, google_api_key, cx_code]):
    raise ValueError("❌ Required API keys are missing! Please check your .env file.")

# --- Load FAISS Vector Store ---
faiss_index_path = "/home/ubuntu/chatbot/faiss_index"  # ✅ Use absolute path
try:
    vectorstore = FAISS.load_local(
        faiss_index_path,
        OpenAIEmbeddings(openai_api_key=openai_api_key),
        allow_dangerous_deserialization=True  # ✅ Allow safe deserialization
    )
except Exception as e:
    raise RuntimeError(f"❌ Failed to load FAISS vector store: {e}")

# --- Set Up Retriever ---
retriever = vectorstore.as_retriever()

print("✅ FAISS Vector Store Loaded Successfully!")
