from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN", "")

print(f"Using model: {MODEL_NAME}")
print(f"API Base: {API_BASE_URL}")