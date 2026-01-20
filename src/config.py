import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    WAHA_API_KEY = os.getenv("WAHA_API_KEY")
    WAHA_BASE_URL = os.getenv("WAHA_BASE_URL", "http://waha:3000")
    DATABASE_URL = os.getenv("DATABASE_URL")
    PDF_PATH = "/app/dados/RECEITASPARARAG.pdf"

settings = Config()