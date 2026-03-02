"""
infrastructure/settings.py — Configurações centralizadas
=========================================================
Única fonte da verdade para variáveis de ambiente da Clean Architecture.

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

    # ── LLM Principal (Google Gemini - Custo Zero) ────────────────────────────
    GEMINI_API_KEY:    str   = ""
    GEMINI_MODEL:      str   = "gemini-2.0-flash" # Modelo super rápido e atualizado
    GEMINI_TEMP:       float = 0.3
    GEMINI_MAX_TOKENS: int   = 1024

    # ── LLM Secundário (Groq - Velocidade / Fallback) ─────────────────────────
    GROQ_API_KEY:    str   = ""
    GROQ_MODEL:      str   = "llama-3.1-8b-instant"

    # ── HuggingFace (Modelos de Embedding Locais) ─────────────────────────────
    HF_TOKEN: str = ""

    # ── RAG / Ingestão ────────────────────────────────────────────────────────
    LLAMA_CLOUD_API_KEY: str = ""
    DATA_DIR:            str = "/app/dados"

    # ── Memória e Vector DB Único (Redis - Alta Performance) ──────────────────
    # O Redis agora faz o papel de Cache, Histórico e Banco de Dados Vetorial (Busca Híbrida)
    REDIS_URL: str = "redis://redis-cache:6379/0"

    # ── WhatsApp (WAHA) ───────────────────────────────────────────────────────
    WAHA_API_KEY:        str = ""
    WAHA_DASHBOARD_PASS: str = ""
    WAHA_BASE_URL:       str = "http://waha:3000"
    WHATSAPP_HOOK_URL:   str = "http://meu-bot:8000/webhook"

    # ── WhatsApp (Evolution API - Alternativa) ────────────────────────────────
    EVOLUTION_BASE_URL:      str = ""
    EVOLUTION_API_KEY:       str = ""
    EVOLUTION_INSTANCE_NAME: str = "default"

    # ── Agente ────────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 6
    AGENT_TIMEOUT_S:      int = 45
    MAX_HISTORY_MESSAGES: int = 8

    # ── Observabilidade (Langfuse) ────────────────────────────────────────────
    # Substitui o LangSmith. Excelente para monitorizar os tokens gastos.
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_BASE_URL:   str = "https://us.cloud.langfuse.com"

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
    def langfuse_ativo(self) -> bool:
        """Verifica se as chaves do Langfuse foram configuradas para ativar o tracing"""
        return bool(self.LANGFUSE_SECRET_KEY and self.LANGFUSE_PUBLIC_KEY)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()