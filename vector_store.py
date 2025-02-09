#!/usr/bin/env python
# coding: utf-8

# In[1]:


# vector_store.py

from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load your data
loader = TextLoader("/chatbot/my_data.txt")  # Ensure this file exists
documents = loader.load()

# Split text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create vector database using OpenAI embeddings
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=api_key))

# Save the FAISS index locally
vectorstore.save_local("faiss_index")
print("Vector database saved successfully!")


# In[ ]:




