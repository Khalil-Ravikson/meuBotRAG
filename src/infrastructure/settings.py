"""
infrastructure/settings.py — Configurações centralizadas v3
============================================================
MIGRAÇÃO v2 → v3:
  REMOVIDO:  GROQ_*, DATABASE_URL, DB_USER/PASS/NAME, WAHA_*, LLAMA_CLOUD_API_KEY
             (o LlamaParse agora é opcional — só necessário se PDF_PARSER=llamaparse)
  ADICIONADO: GEMINI_*, REDIS_URL (Redis Stack substitui pgvector),
              PDF_PARSER (escolha do parser de PDF), LLAMA_CLOUD_API_KEY (opcional)
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE_PATH", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gemini (substitui Groq) ───────────────────────────────────────────────
    GEMINI_API_KEY:    str   = ""
    GEMINI_MODEL:      str   = "gemini-2.0-flash"
    GEMINI_TEMP:       float = 0.3
    GEMINI_MAX_TOKENS: int   = 1024

    # ── HuggingFace (para BAAI/bge-m3) ───────────────────────────────────────
    HF_TOKEN: str = ""

    # ── Parser de PDF ─────────────────────────────────────────────────────────
    # Escolhe qual parser usar para extrair texto dos PDFs.
    #
    # "pymupdf"    → local, gratuito, rápido (~50ms/pág). Default.
    #                Bom para PDFs semi-estruturados.
    #                Requer: pip install pymupdf
    #
    # "llamaparse" → cloud, pago (~$0.003/pág), melhor para tabelas complexas.
    #                Requer: pip install llama-parse  +  LLAMA_CLOUD_API_KEY abaixo
    #
    # Podes também forçar por ficheiro individual no PDF_CONFIG (ingestion.py):
    #   "edital_paes_2026.pdf": { ..., "parser": "llamaparse" }
    PDF_PARSER: str = "pymupdf"

    # ── LlamaParse (só necessário se PDF_PARSER=llamaparse) ───────────────────
    # Obtém em: https://cloud.llamaindex.ai
    LLAMA_CLOUD_API_KEY: str = ""

    # ── RAG ───────────────────────────────────────────────────────────────────
    DATA_DIR: str = "/app/dados"

    # ── Redis Stack ───────────────────────────────────────────────────────────
    # Substitui pgvector + redis:alpine da v2.
    # DB 0 → bot (chunks, memória, vectores)
    # DB 1 → Evolution API (cache de sessões)
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Evolution API (WhatsApp) ──────────────────────────────────────────────
    EVOLUTION_BASE_URL:      str = "http://localhost:8080"
    EVOLUTION_API_KEY:       str = ""
    EVOLUTION_INSTANCE_NAME: str = "default"
    WHATSAPP_HOOK_URL:       str = "http://bot:9000/webhook"

    # ── Agente ────────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 6
    AGENT_TIMEOUT_S:      int = 45
    MAX_HISTORY_MESSAGES: int = 8

    # ── Semantic Router ───────────────────────────────────────────────────────
    # Limiar de similaridade para o routing semântico.
    # 0.0 = sempre usa a tool mais próxima (mesmo que pouco similar)
    # 0.5 = só usa tool se a similaridade for >= 50% (mais conservador)
    ROUTER_SIMILARITY_THRESHOLD: float = 0.35

    # ── LangSmith (opcional) ──────────────────────────────────────────────────
    LANGCHAIN_API_KEY:    str  = ""
    LANGCHAIN_PROJECT:    str  = "uema-bot-v3"
    LANGCHAIN_TRACING_V2: bool = False

    # ── Dev / Debug ───────────────────────────────────────────────────────────
    DEV_MODE:      bool = False
    DEV_WHITELIST: str  = ""
    LOG_LEVEL:     str  = "INFO"

    # ── Propriedades derivadas ────────────────────────────────────────────────

    @property
    def dev_whitelist_list(self) -> list[str]:
        if not self.DEV_WHITELIST:
            return []
        return [n.strip() for n in self.DEV_WHITELIST.split(",") if n.strip()]

    @property
    def langsmith_ativo(self) -> bool:
        return bool(self.LANGCHAIN_API_KEY and self.LANGCHAIN_TRACING_V2)

    @property
    def llamaparse_disponivel(self) -> bool:
        """True se o LlamaParse está configurado e pronto a usar."""
        return bool(self.LLAMA_CLOUD_API_KEY) and self.PDF_PARSER.lower() == "llamaparse"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()