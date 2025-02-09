#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import yfinance as yf
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # ✅ Use the correct import
from langchain.schema import Document

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Check your .env file.")

# Load FAISS index
vector_store_path = "faiss_index"
embedding_function = OpenAIEmbeddings(api_key=openai_api_key)
vector_store = FAISS.load_local(
    folder_path=vector_store_path,
    embeddings=embedding_function,
    allow_dangerous_deserialization=True
)

# ✅ Fetch live SBI stock price
sbi = yf.Ticker("SBIN.NS")  # NSE India ticker for SBI
sbi_price = sbi.history(period="1d")["Close"].iloc[-1]
print(f"\n✅ Fetched SBI Share Price: ₹{sbi_price}")

# ✅ Prepare document for FAISS
sbi_doc = Document(page_content=f"SBI Share Price: ₹{sbi_price}", metadata={"source": "yfinance"})

# ✅ Add to FAISS index and save
vector_store.add_documents([sbi_doc])
vector_store.save_local(vector_store_path)
print("\n✅ FAISS index updated successfully!")


# In[ ]:




