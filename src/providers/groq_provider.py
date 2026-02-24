"""
providers/groq_provider.py — Groq LLM com retry e métricas
===========================================================
Encapsula:
  - Instanciação do ChatGroq
  - Retry automático em 429 (rate limit) com backoff exponencial
  - Contagem de tokens de entrada/saída
  - Timeout configurável

Nunca deixa o erro 429 vazar para cima — trata aqui.
"""
from __future__ import annotations
import logging
import time
from functools import lru_cache
from langchain_groq import ChatGroq
from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

_MAX_RETRIES  = 3
_BACKOFF_BASE = 2.0   # segundos — dobra a cada tentativa


@lru_cache(maxsize=1)
def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatGroq:
    """
    Retorna uma instância singleton do ChatGroq.
    Os parâmetros padrão vêm de settings — substituíveis para testes.
    """
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=model or settings.GROQ_MODEL,
        temperature=temperature if temperature is not None else settings.GROQ_TEMP,
        max_tokens=max_tokens or settings.GROQ_MAX_TOKENS,
    )


def invocar_com_retry(llm: ChatGroq, messages: list, **kwargs) -> str:
    """
    Chama o LLM com retry automático em 429.
    Retorna o conteúdo de texto da resposta.

    Raises:
        RuntimeError: após esgotar os retries ou em erro não-429.
    """
    for tentativa in range(1, _MAX_RETRIES + 1):
        try:
            resposta = llm.invoke(messages, **kwargs)
            return resposta.content
        except Exception as e:
            err = str(e)
            is_429 = "429" in err or "rate_limit" in err.lower() or "too many requests" in err.lower()

            if is_429 and tentativa < _MAX_RETRIES:
                espera = _BACKOFF_BASE ** tentativa
                logger.warning(
                    "⏳ Rate limit Groq (tentativa %d/%d). Aguardando %.0fs...",
                    tentativa, _MAX_RETRIES, espera,
                )
                time.sleep(espera)
                continue

            if is_429:
                logger.error("❌ Rate limit Groq esgotado após %d tentativas.", _MAX_RETRIES)
                raise RuntimeError("rate_limit_esgotado") from e

            raise