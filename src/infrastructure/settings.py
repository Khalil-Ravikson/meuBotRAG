"""
infrastructure/settings.py — Configurações v5
===============================================
NOVIDADES v5:
  ADICIONADO:
    ADMIN_NUMBERS   → lista de admins RBAC (vírgula separado)
    STUDENT_NUMBERS → lista de students RBAC (opcional)
    ADMIN_API_KEY   → chave para endpoints REST admin
    WEBHOOK_SECRET  → valida que webhook veio da Evolution API
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file          = os.getenv("ENV_FILE_PATH", ".env"),
        env_file_encoding = "utf-8",
        case_sensitive    = False,
        extra             = "ignore",
    )

    # ── Gemini ────────────────────────────────────────────────────────────────
    GEMINI_API_KEY:    str   = ""
    GEMINI_MODEL:      str   = "gemini-2.0-flash"
    GEMINI_TEMP:       float = 0.3
    GEMINI_MAX_TOKENS: int   = 1024

    # ── HuggingFace ───────────────────────────────────────────────────────────
    HF_TOKEN: str = ""

    # ── Parser de PDF ─────────────────────────────────────────────────────────
    PDF_PARSER:          str = "pymupdf"
    LLAMA_CLOUD_API_KEY: str = ""

    # ── RAG ───────────────────────────────────────────────────────────────────
    DATA_DIR: str = "/app/dados"

    # ── Redis Stack ───────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Evolution API ─────────────────────────────────────────────────────────
    EVOLUTION_BASE_URL:      str = "http://localhost:8080"
    EVOLUTION_API_KEY:       str = ""
    EVOLUTION_INSTANCE_NAME: str = "default"
    WHATSAPP_HOOK_URL:       str = "http://bot:9000/webhook"

    # ── Segurança do Webhook ─────────────────────────────────────────────────
    # Chave enviada pela Evolution API no header "apikey" do webhook
    # Configurar em Evolution → Webhook → Headers: apikey = WEBHOOK_SECRET
    WEBHOOK_SECRET: str = ""

    # ── RBAC (NOVO v5) ────────────────────────────────────────────────────────
    # Números de WhatsApp separados por vírgula (sem + nem espaço)
    # Ex: "5598999990001,5598999990002"
    ADMIN_NUMBERS:   str = ""   # ADMIN: acesso total + comandos admin
    STUDENT_NUMBERS: str = ""   # STUDENT: RAG + GLPI (se vazio, todos são GUEST)
    ADMIN_API_KEY:   str = ""   # Chave para endpoints REST /admin/*

    # ── Agente ────────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 6
    AGENT_TIMEOUT_S:      int = 45
    MAX_HISTORY_MESSAGES: int = 8

    # ── Semantic Router ───────────────────────────────────────────────────────
    ROUTER_SIMILARITY_THRESHOLD: float = 0.35

    # ── LangSmith (opcional) ──────────────────────────────────────────────────
    LANGCHAIN_API_KEY:    str  = ""
    LANGCHAIN_PROJECT:    str  = "uema-bot-v5"
    LANGCHAIN_TRACING_V2: bool = False

    # ── Dev / Debug ───────────────────────────────────────────────────────────
    DEV_MODE:      bool = False
    DEV_WHITELIST: str  = ""
    LOG_LEVEL:     str  = "INFO"

    # ── Propriedades derivadas ────────────────────────────────────────────────

    @property
    def dev_whitelist_list(self) -> list[str]:
        return [n.strip() for n in self.DEV_WHITELIST.split(",") if n.strip()]

    @property
    def admin_list(self) -> list[str]:
        return [n.strip() for n in self.ADMIN_NUMBERS.split(",") if n.strip()]

    @property
    def student_list(self) -> list[str]:
        return [n.strip() for n in self.STUDENT_NUMBERS.split(",") if n.strip()]

    @property
    def langsmith_ativo(self) -> bool:
        return bool(self.LANGCHAIN_API_KEY and self.LANGCHAIN_TRACING_V2)

    @property
    def llamaparse_disponivel(self) -> bool:
        return bool(self.LLAMA_CLOUD_API_KEY) and self.PDF_PARSER.lower() == "llamaparse"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()