#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv, dotenv_values

# ✅ Absolute path to the .env file
env_path = "/home/ubuntu/chatbot/.env"

# ✅ Load existing environment variables
load_dotenv(env_path)

def update_env_variable(key, value):
    """Update or add an environment variable in the .env file."""
    env_vars = dotenv_values(env_path)  # Load existing .env values
    env_vars[key] = value  # Update the key with the new value

    # ✅ Write updated variables back to the .env file
    with open(env_path, "w") as f:
        for k, v in env_vars.items():
            f.write(f"{k}={v}\n")

    print(f"✅ Updated {key} in .env file")

# ✅ Example: Updating API keys
update_env_variable("OPENAI_API_KEY", "sk-new-openai-key")
update_env_variable("GOOGLE_API_KEY", "new-google-api-key")
update_env_variable("GOOGLE_CX_CODE", "new-google-cx-code")

# ✅ Reload environment variables after update
load_dotenv(env_path)

# ✅ Verify the changes
print(f"🔹 OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:5]}...")  # Masked for security
print(f"🔹 GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')[:5]}...")
print(f"🔹 GOOGLE_CX_CODE: {os.getenv('GOOGLE_CX_CODE')[:5]}...")
