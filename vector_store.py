#!/usr/bin/env python
# coding: utf-8
"""
vector_store.py
---------------
This script:
- Loads documents from a text file.
- Splits them into smaller chunks.
- Creates embeddings using OpenAIEmbeddings.
- Builds a FAISS vector store from those embeddings.
- Saves the FAISS index locally.
API keys are loaded from a .env file. Make sure to add .env to your .gitignore.
"""

from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Load Environment Variables ---
env_path = "/home/ubuntu/chatbot/.env"
load_dotenv(env_path)  # This loads variables from .env in the current directory

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

# --- Load Your Data ---
data_path = "/home/ubuntu/chatbot/my_data.txt"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File {data_path} not found. Please check the path.")

loader = TextLoader(data_path)
documents = loader.load()

# --- Split Documents into Smaller Chunks ---
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# --- Create FAISS Vector Store Using OpenAI Embeddings ---
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=api_key))

# --- Save the FAISS Index Locally ---
vectorstore.save_local("faiss_index")
print("✅ Vector database saved successfully!")
