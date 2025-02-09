#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# --- Load Environment Variables ---
env_path = "/home/ubuntu/chatbot/.env"  # ✅ Set absolute path
load_dotenv(env_path)  
api_key = os.getenv("OPENAI_API_KEY")

# --- Ensure API Key is Loaded ---
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing! Please check your .env file.")

# --- Load FAISS Vector Store ---
faiss_index_path = "/home/ubuntu/chatbot/faiss_index"  # ✅ Use absolute path
vectorstore = FAISS.load_local(
    faiss_index_path,
    OpenAIEmbeddings(openai_api_key=api_key),
    allow_dangerous_deserialization=True  # ✅ Allow safe deserialization
)

# --- Set Up Retriever ---
retriever = vectorstore.as_retriever()

print("✅ FAISS Vector Store Loaded Successfully!")
