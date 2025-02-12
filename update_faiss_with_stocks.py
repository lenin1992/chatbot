#!/usr/bin/env python
# coding: utf-8

import os
import yfinance as yf
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS  # ✅ Corrected import
from langchain_openai import OpenAIEmbeddings  # ✅ Corrected import
from langchain.schema import Document

# ✅ Load Environment Variables
env_path = "/home/ubuntu/chatbot/.env"  # Absolute path to .env file
load_dotenv(env_path)
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Ensure API Key is Loaded
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Check your .env file.")

# ✅ Load FAISS Index
vector_store_path = "/home/ubuntu/chatbot/faiss_index"  # Absolute path for consistency
embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)  # ✅ Pass API key correctly
vector_store = FAISS.load_local(
    folder_path=vector_store_path,
    embeddings=embedding_function,
    allow_dangerous_deserialization=True  # ✅ Allow safe deserialization
)

# ✅ Fetch Live SBI Stock Price
sbi = yf.Ticker("SBIN.NS")  # NSE India ticker for SBI
sbi_price = sbi.history(period="1d")["Close"].iloc[-1]
print(f"\n✅ Fetched SBI Share Price: ₹{sbi_price}")

# ✅ Prepare Document for FAISS
sbi_doc = Document(page_content=f"SBI Share Price: ₹{sbi_price}", metadata={"source": "yfinance"})

# ✅ Add to FAISS Index and Save
vector_store.add_documents([sbi_doc])
vector_store.save_local(vector_store_path)
print("\n✅ FAISS index updated successfully!")
