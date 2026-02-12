import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    WAHA_API_KEY = os.getenv("WAHA_API_KEY")
    WAHA_BASE_URL = os.getenv("WAHA_BASE_URL")
    DATABASE_URL = os.getenv("DATABASE_URL")
    WHATSAPP_HOOK_URL = os.getenv("WHATSAPP_HOOK_URL")
    PDF_PATH = os.getenv("PDF_PATH", "/app/dados/calendario-academico-2026.pdf")
    REDIS_URL = os.getenv("REDIS_URL")
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
settings = Config()