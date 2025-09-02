import os
from dotenv import load_dotenv

def get_api_key():
    load_dotenv()
    for key in ["OPENAI_API_KEY","ANTHROPIC_API_KEY","GEMINI_API_KEY"]:
        v = os.getenv(key)
        if v:
            return key, v
    return None, None

def bullet(items):
    return "\n".join([f"- {x}" for x in items])
