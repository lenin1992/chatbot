#!/usr/bin/env python
# coding: utf-8

# In[2]:


from dotenv import load_dotenv
import os
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS Vector Store
vectorstore = FAISS.load_local(
    "faiss_index",
    OpenAIEmbeddings(openai_api_key=api_key),
    allow_dangerous_deserialization=True  # ✅ Allow safe deserialization
)


# Set up retriever
retriever = vectorstore.as_retriever()


# In[ ]:




