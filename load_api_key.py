#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")

# Print API key (so subprocess in retrieval_qa.py can read it)
if api_key:
    print(api_key)  # ✅ This is required for retrieval_qa.py
else:
    print("Error: OPENAI_API_KEY is missing.")
