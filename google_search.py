#!/usr/bin/env python
# coding: utf-8

# In[18]:





# In[23]:


import os
import requests
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ✅ Set API Credentials
os.environ["OPENAI_API_KEY"] = "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"
API_KEY = "AIzaSyDor0Ki9yEhlPucvQW5pXgn8AwXOv_pqxw"
CX_CODE = "7182a7023f44f47cd"

# ✅ Function to Fetch Google Search Results
def fetch_google_results(query):
    """Fetch top 10 search results from Google Custom Search API."""
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX_CODE}"
    response = requests.get(url)
    data = response.json()
    
    results = []
    for item in data.get("items", []):
        text = f"{item['title']} - {item['snippet']} ({item['link']})"
        results.append(Document(page_content=text))
    
    return results

# ✅ Function to Update FAISS Index with Google Search Data
def update_faiss_with_google(query, faiss_index_path="faiss_index"):
    """Fetch Google results and store them in FAISS."""
    
    # Load existing FAISS index (or create a new one)
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Fetch Google Search results
    google_docs = fetch_google_results(query)
    
    # Split into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(google_docs)
    
    # Add to FAISS
    vectorstore.add_documents(chunks)
    
    # Save updated FAISS index
    vectorstore.save_local(faiss_index_path)
    print(f"🔹 Google search data added to FAISS for query: {query}")

# ✅ Function to Retrieve Relevant Documents from FAISS
def retrieve_from_faiss(query, faiss_index_path="faiss_index"):
    """Retrieve relevant documents from FAISS."""
    
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)

    
    # Print Retrieved Documents
    for i, doc in enumerate(retrieved_docs):
        print(f"\n🔹 Retrieved Document {i+1}:\n{doc.page_content[:500]}")

# ✅ Example Usage:
query = "latest AI trends 2025"
update_faiss_with_google(query)  # Update FAISS with new search results
retrieve_from_faiss(query)       # Retrieve data from FAISS


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




