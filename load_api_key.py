from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file.")

print(f"Loaded API Key: {api_key[:5]}...")  # Partial print for verification
