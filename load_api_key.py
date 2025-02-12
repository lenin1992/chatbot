from dotenv import load_dotenv
import os

# ✅ Corrected Absolute Path
env_path = "/home/ubuntu/chatbot/.env"  # Added leading slash
load_dotenv(env_path)  # Load from .env

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file.")

print(f"Loaded API Key: {api_key[:5]}...")  # Partial print for verification
