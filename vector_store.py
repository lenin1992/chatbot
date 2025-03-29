#!/usr/bin/env python
# coding: utf-8
"""
vector.py
---------------
This script:
- Loads documents from a text file.
- Splits them into smaller chunks.
- Creates embeddings using OpenAIEmbeddings.
- Builds a FAISS vector store from those embeddings.
- Saves the FAISS index locally.

⚠️ Ensure your .env file contains the OPENAI_API_KEY.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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

# ✅ Load Data
data_path = "/home/ubuntu/chatbot/my_data.txt"
if not os.path.exists(data_path):
    logging.error(f"❌ Data file not found: {data_path}")
    raise FileNotFoundError(f"❌ File {data_path} not found. Please check the path.")

logging.info("📄 Loading documents...")
loader = TextLoader(data_path)
documents = loader.load()

# ✅ Split Documents
logging.info("🔹 Splitting documents into smaller chunks...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# ✅ Create FAISS Vector Store
logging.info("📌 Generating vector embeddings using OpenAI API...")
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))

# ✅ Save FAISS Index
faiss_index_path = "/home/ubuntu/chatbot/faiss_index"
vectorstore.save_local(faiss_index_path)
logging.info("✅ Vector database saved successfully at: " + faiss_index_path)

if __name__ == "__main__":
    logging.info("🚀 FAISS Vector Store Generation Completed!")
