#!/usr/bin/env python
# coding: utf-8
"""
update.py
---------------
This script:
- Loads the FAISS vector store.
- Fetches the latest SBI stock price from Yahoo Finance.
- Updates the FAISS index with the new data.
- Saves the updated FAISS index locally.

⚠️ Ensure your .env file contains the OPENAI_API_KEY.
"""

import os
import logging
import yfinance as yf
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

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

# ✅ Load FAISS Index
vector_store_path = "/home/ubuntu/chatbot/faiss_index"
embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)

try:
    vector_store = FAISS.load_local(
        folder_path=vector_store_path,
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )
    logging.info("✅ FAISS vector store loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load FAISS index: {e}")
    raise RuntimeError("❌ FAISS index loading failed. Check file path and integrity.")

# ✅ Fetch Live SBI Stock Price
try:
    sbi = yf.Ticker("SBIN.NS")
    sbi_price = sbi.history(period="1d")["Close"].iloc[-1]
    logging.info(f"📈 SBI Share Price Fetched: ₹{sbi_price}")
except Exception as e:
    logging.error(f"❌ Failed to fetch SBI stock price: {e}")
    raise RuntimeError("❌ Unable to fetch stock price. Check internet connection or ticker symbol.")

# ✅ Prepare Document for FAISS
sbi_doc = Document(page_content=f"SBI Share Price: ₹{sbi_price}", metadata={"source": "yfinance"})

# ✅ Add to FAISS Index and Save
vector_store.add_documents([sbi_doc])
vector_store.save_local(vector_store_path)
logging.info("✅ FAISS index updated successfully.")

if __name__ == "__main__":
    logging.info("🚀 FAISS Vector Store Update Completed!")
