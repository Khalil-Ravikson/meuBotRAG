"""
infrastructure/settings.py — Configurações centralizadas
=========================================================
Única fonte da verdade para variáveis de ambiente.

Importe em qualquer lugar:
    from src.infrastructure.settings import settings
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE_PATH", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (Groq) ────────────────────────────────────────────────────────────
    GROQ_API_KEY:    str   = ""
    GROQ_MODEL:      str   = "llama-3.1-8b-instant"
    GROQ_TEMP:       float = 0.3
    GROQ_MAX_TOKENS: int   = 1024

    # ── HuggingFace ───────────────────────────────────────────────────────────
    # HF_TOKEN acelera o DOWNLOAD do modelo (evita rate limit do Hub).
    # Não afeta a velocidade de inferência (isso depende de CPU/GPU).
    # Obtenha em: https://huggingface.co/settings/tokens
    HF_TOKEN: str = ""

    # ── RAG / Ingestão ────────────────────────────────────────────────────────
    LLAMA_CLOUD_API_KEY: str = ""
    DATA_DIR:            str = "/app/dados"

    # ── Banco vetorial (pgvector) ─────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/vectordb"

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── WAHA (DESCONTINUADO) ───────────────────────────────────────────────────────
    # WAHA_API_KEY:      str = ""
    # WAHA_BASE_URL:     str = "http://waha:3000"
    # WAHA_SESSION:      str = "default"
    # WHATSAPP_HOOK_URL: str = "http://bot-rag:8000/webhook"

    # ── EVOLUTION (WhatsApp) ───────────────────────────────────────────────────────
    EVOLUTION_BASE_URL: str
    EVOLUTION_API_KEY: str
    EVOLUTION_INSTANCE_NAME: str = "default"
    WHATSAPP_HOOK_URL: str = "http://bot-rag:8000/webhook"
    # ── Agente ────────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 6
    AGENT_TIMEOUT_S:      int = 45
    MAX_HISTORY_MESSAGES: int = 8

    # ── LangSmith (observabilidade LangChain) ─────────────────────────────────
    # Rastreia chamadas do agente, tokens, tools, latência no dashboard.
    # Obtenha em: https://smith.langchain.com → Settings → API Keys
    # Quando vazio, o LangSmith fica desativado automaticamente.
    LANGCHAIN_API_KEY:    str  = ""
    LANGCHAIN_PROJECT:    str  = "uema-bot"
    LANGCHAIN_TRACING_V2: bool = False   # True ativa o rastreamento

    # ── Dev / Debug ───────────────────────────────────────────────────────────
    DEV_MODE:      bool = False
    DEV_WHITELIST: str  = ""
    LOG_LEVEL:     str  = "INFO"

    @property
    def dev_whitelist_list(self) -> list[str]:
        if not self.DEV_WHITELIST:
            return []
        return [n.strip() for n in self.DEV_WHITELIST.split(",") if n.strip()]

    @property
    def langsmith_ativo(self) -> bool:
        return bool(self.LANGCHAIN_API_KEY and self.LANGCHAIN_TRACING_V2)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()