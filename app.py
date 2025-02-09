#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests

# --- Set API Keys ---
os.environ["OPENAI_API_KEY"] = "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"
API_KEY = "AIzaSyDor0Ki9yEhlPucvQW5pXgn8AwXOv_pqxw"
CX_CODE = "7182a7023f44f47cd"

# --- Load FAISS Index ---
faiss_index_path = "faiss_index"
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# --- Function to Retrieve from FAISS ---
def retrieve_faiss_results(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Get top 3 results
    retrieved_docs = retriever.invoke(query)

    # 🔹 **Check Relevance**: Ignore FAISS results if they seem irrelevant
    relevant_docs = []
    for doc in retrieved_docs:
        if query.lower() in doc.page_content.lower():  # Simple relevance check
            relevant_docs.append(doc)

    return relevant_docs

# --- Function to Fetch Google Search Results ---
def fetch_google_results(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX_CODE}"
    response = requests.get(url)
    data = response.json()
    
    results = []
    for item in data.get("items", []):
        text = f"{item['title']} - {item['snippet']} ({item['link']})"
        results.append(Document(page_content=text))
    
    return results

# --- Function to Handle Small Talk Queries ---
def handle_small_talk(query):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if query.lower() in greetings:
        return "👋 Hello! How can I assist you today?"
    return None  # If it's not small talk, return None

# --- Streamlit UI ---
st.set_page_config(page_title="AI Chatbot with Google Search & FAISS", layout="wide")
st.title("🤖 AI Chatbot with Google Search & FAISS")

# --- Input Box ---
query = st.text_input("🔍 Ask something...", placeholder="e.g., What are the latest AI trends in 2025?")

if st.button("Search & Generate Answer"):
    if query:
        # 🔹 **Check for Small Talk**
        small_talk_response = handle_small_talk(query)
        if small_talk_response:
            st.success(small_talk_response)  # ✅ Return friendly response
        else:
            with st.status("⏳ **Fetching results... Please wait.**", expanded=True):
                # --- Retrieve from FAISS ---
                retrieved_docs = retrieve_faiss_results(query)

                # --- Fetch Google Search Results & Update FAISS ---
                google_docs = fetch_google_results(query)
                vectorstore.add_documents(google_docs)
                vectorstore.save_local(faiss_index_path)

                # --- Display Retrieved Results ---
                if retrieved_docs:
                    st.subheader("🔹 Relevant Documents from FAISS:")
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"**{i+1}.** {doc.page_content}")

                st.subheader("🌐 Top Google Search Results:")
                for i, doc in enumerate(google_docs[:3]):  # Show top 3 results
                    st.write(f"**{i+1}.** {doc.page_content}")

            st.success("✅ Results fetched successfully!")
    else:
        st.warning("⚠️ Please enter a query.")

# --- Footer ---
st.markdown("<br><center>🔹 Made with ❤️ by AI Enthusiast</center>", unsafe_allow_html=True)


# In[ ]:




