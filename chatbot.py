#!/usr/bin/env python
# coding: utf-8

import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Load Environment Variables
env_path = "/home/ubuntu/chatbot/.env"
if not os.path.exists(env_path):
    logging.error("❌ .env file not found! Check the path.")
    raise FileNotFoundError("❌ .env file missing. Please ensure it's in the correct directory.")

load_dotenv(env_path)

# ✅ Retrieve API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")  # 🔹 Fixed key name
cx_code = os.getenv("GOOGLE_CX_CODE")  # 🔹 Fixed key name

# ✅ Ensure All API Keys Are Loaded
if not all([openai_api_key, google_api_key, cx_code]):
    logging.error("❌ One or more API keys are missing! Check your .env file.")
    raise ValueError("❌ Required API keys are missing! Please update your .env file.")

# ✅ Load FAISS Vector Store
faiss_index_path = "/home/ubuntu/chatbot/faiss_index"

try:
    vectorstore = FAISS.load_local(
        faiss_index_path,
        OpenAIEmbeddings(openai_api_key=openai_api_key),
        allow_dangerous_deserialization=True  # ✅ Allow controlled deserialization
    )
    logging.info("✅ FAISS Vector Store Loaded Successfully!")
except Exception as e:
    logging.error(f"❌ Failed to load FAISS vector store: {e}")
    raise RuntimeError("❌ FAISS vector store could not be loaded.")

# ✅ Set Up Retriever
retriever = vectorstore.as_retriever()

if __name__ == "__main__":
    logging.info("🚀 Chatbot is ready! You can now use the retriever.")
