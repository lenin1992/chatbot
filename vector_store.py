#!/usr/bin/env python
# coding: utf-8

# vector_store.py

from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is set
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your environment variables.")

# Load your data
data_path = "/home/ubuntu/chatbot/my_data.txt"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File {data_path} not found. Please check the path.")

loader = TextLoader(data_path)
documents = loader.load()

# Split text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create vector database using OpenAI embeddings
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=api_key))

# Save the FAISS index locally
vectorstore.save_local("faiss_index")
print("✅ Vector database saved successfully!")
