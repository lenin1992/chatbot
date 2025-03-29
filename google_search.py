#!/usr/bin/env python
# coding: utf-8

import os
import logging
from dotenv import load_dotenv, dotenv_values

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Absolute Path to the .env File
env_path = "/home/ubuntu/chatbot/.env"

# ✅ Check if .env file exists
if not os.path.exists(env_path):
    logging.error("❌ .env file not found! Ensure the correct path.")
    raise FileNotFoundError("❌ .env file missing. Please create it in the correct directory.")

# ✅ Load Environment Variables
load_dotenv(env_path)

def update_env_variable(key, value):
    """Update or add an environment variable in the .env file."""
    env_vars = dotenv_values(env_path)  # Load existing values

    if key in env_vars and env_vars[key] == value:
        logging.info(f"✅ {key} is already up to date.")
        return

    env_vars[key] = value  # Update the key with the new value

    # ✅ Write updated variables back to the .env file
    try:
        with open(env_path, "w") as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")

        logging.info(f"✅ Successfully updated {key} in .env file.")
    except Exception as e:
        logging.error(f"❌ Failed to update {key}: {e}")
        raise RuntimeError(f"❌ Error writing to .env file: {e}")

# ✅ Example: Updating API Keys
api_keys = {
    "OPENAI_API_KEY": "sk-new-openai-key",
    "GOOGLE_API_KEY": "new-google-api-key",
    "GOOGLE_CX_CODE": "new-google-cx-code"
}

for key, value in api_keys.items():
    update_env_variable(key, value)

# ✅ Reload Environment Variables After Update
load_dotenv(env_path)

# ✅ Verify the Changes (Masked Output for Security)
logging.info(f"🔹 OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:5]}...")
logging.info(f"🔹 GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')[:5]}...")
logging.info(f"🔹 GOOGLE_CX_CODE: {os.getenv('GOOGLE_CX_CODE')[:5]}...")

if __name__ == "__main__":
    logging.info("🚀 Google Search API Configuration Updated!")
