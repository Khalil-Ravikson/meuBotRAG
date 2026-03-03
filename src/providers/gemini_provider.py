"""
providers/gemini_provider.py — Gemini Flash com Tenacity Resiliente (V4 — Corrigido)
======================================================================================

BUGS CORRIGIDOS vs versão anterior:
─────────────────────────────────────
  BUG 1 (CRÍTICO — TypeError silencioso): chamar_gemini_estruturado() aceitava
    `schema_descricao: str` em query_transform.py, mas a assinatura esperava
    `response_schema: type[BaseModel]`. Resultado: query transform sempre caía
    no fallback sem lançar erro visível.
    CORRIGIDO: chamar_gemini_estruturado() aceita AMBOS os formatos:
      - response_schema (Pydantic) → usa Structured Output nativo da API
      - schema_descricao (str)     → injeta no prompt como instrução JSON
    Mantém compatibilidade com query_transform.py SEM obrigar refactor imediato.

  BUG 2 (CRÍTICO — Retry cego): @retry sem retry_if_exception fazia retry
    em QUALQUER exceção, incluindo INVALID_ARGUMENT, API_KEY_INVALID, etc.
    Desperdiçava tentativas e mascarava bugs reais de configuração.
    CORRIGIDO: retry_if_exception(_is_retryable) filtra apenas 429/503/timeout.

  BUG 3 (Menor — Duplicação): SYSTEM_UEMA e montar_prompt_geracao estavam
    duplicados em gemini_provider.py e agent/prompts.py.
    CORRIGIDO: gemini_provider.py importa de agent/prompts.py.
    agent/prompts.py é a fonte única de verdade dos prompts.
    (Mantemos re-export de SYSTEM_UEMA e montar_prompt_geracao aqui para
    não quebrar o import em agent/core.py.)
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import google.genai as genai
from google.genai import types
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.infrastructure.settings import settings

# Importa prompts de agent/prompts.py — FONTE ÚNICA DE VERDADE
# Re-exportamos aqui para manter compatibilidade com imports existentes em core.py
from src.agent.prompts import (
    SYSTEM_UEMA,
    montar_prompt_geracao,
    PROMPT_QUERY_REWRITE,
    PROMPT_EXTRACAO_FATOS,
)

__all__ = [
    "SYSTEM_UEMA",
    "montar_prompt_geracao",
    "PROMPT_QUERY_REWRITE",
    "PROMPT_EXTRACAO_FATOS",
    "GeminiResponse",
    "get_gemini_client",
    "chamar_gemini",
    "chamar_gemini_async",
    "chamar_gemini_estruturado",
    "QueryRewriteSchema",
    "ExtracaoFatosSchema",
]

logger = logging.getLogger(__name__)

# Modelos disponíveis (fallback automático se o primário saturar)
MODELO_PRIMARIO = "gemini-2.5-flash"
MODELO_FALLBACK = "gemini-2.0-flash-lite"


# ─────────────────────────────────────────────────────────────────────────────
# Schemas Pydantic para Structured Outputs nativos
# ─────────────────────────────────────────────────────────────────────────────

class QueryRewriteSchema(BaseModel):
    query_reescrita: str
    palavras_chave: list[str]

class ExtracaoFatosSchema(BaseModel):
    fatos: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Response normalizado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeminiResponse:
    conteudo:      str
    model:         str
    input_tokens:  int  = 0
    output_tokens: int  = 0
    sucesso:       bool = True
    erro:          str  = ""

    @property
    def tokens_total(self) -> int:
        return self.input_tokens + self.output_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Cliente singleton
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY não configurada no .env")
    client = genai.Client(api_key=api_key)
    logger.info("✅ Cliente Gemini inicializado | modelo=%s", MODELO_PRIMARIO)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Filtro de erros retriáveis — CORRIGIDO
# ─────────────────────────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """
    Decide se o Tenacity deve fazer retry neste erro.

    RETRIÁVEIS (problemas temporários de infra/quota):
      - 429 Too Many Requests (Rate Limit)
      - 503 Service Unavailable (modelo sobrecarregado)
      - Timeout de rede (conexão interrompida)

    NÃO RETRIÁVEIS (bugs de código ou configuração):
      - 400 INVALID_ARGUMENT  → prompt malformado, schema errado
      - 401 / 403             → API key inválida ou sem permissão
      - 404                   → modelo não existe
      - Qualquer outro erro   → não adianta tentar de novo

    Por que isto importa?
      Sem este filtro, o Tenacity faz 4 tentativas num INVALID_ARGUMENT,
      gastando 30s+ e mascarando o bug real. Com o filtro, falha na 1ª
      tentativa e o erro aparece imediatamente nos logs.
    """
    err = str(exc).lower()
    return (
        "429"         in err or
        "quota"       in err or
        "rate limit"  in err or
        "503"         in err or
        "overloaded"  in err or
        "timeout"     in err or
        "connection"  in err
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chamada core com Tenacity — CORRIGIDO
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    # Filtra: só faz retry em erros temporários (429, 503, timeout)
    retry=retry_if_exception(_is_retryable),

    # Backoff exponencial: 2s → 4s → 8s → 16s (4 tentativas = ~30s total)
    # Adequado para o Free Tier do Gemini que tem janelas de 1 minuto
    wait=wait_exponential(multiplier=2, min=2, max=16),

    # Para depois de 4 tentativas. Se o rate limit persistir, o Celery
    # fará retry da task inteira (com backoff maior)
    stop=stop_after_attempt(4),

    # Loga cada tentativa para diagnóstico
    before_sleep=before_sleep_log(logger, logging.WARNING),

    # IMPORTANTE: reraise=True → após esgotar tentativas, lança a exceção
    # original para que chamar_gemini() possa tratar o fallback
    reraise=True,
)
def _chamar_gemini_com_retry(
    prompt: str,
    system_instruction: str | None,
    temperatura: float,
    max_tokens: int,
    modelo: str,
    response_schema: type[BaseModel] | None,
) -> GeminiResponse:
    """
    Chamada síncrona ao Gemini protegida por retry inteligente.
    Não chamar diretamente — usar chamar_gemini() como interface pública.
    """
    client = get_gemini_client()

    config_kwargs: dict[str, Any] = {
        "temperature":      temperatura,
        "max_output_tokens": max_tokens,
        "safety_settings": [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ],
    }

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    # Structured Output nativo (quando fornecido schema Pydantic)
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"]    = response_schema

    config = types.GenerateContentConfig(**config_kwargs)

    resposta = client.models.generate_content(
        model=modelo,
        contents=prompt,
        config=config,
    )

    usage      = getattr(resposta, "usage_metadata", None)
    input_tok  = getattr(usage, "prompt_token_count",      0) if usage else 0
    output_tok = getattr(usage, "candidates_token_count",  0) if usage else 0
    conteudo   = resposta.text or ""

    logger.debug(
        "✅ Gemini | modelo=%s | in=%d | out=%d | chars=%d",
        modelo, input_tok, output_tok, len(conteudo),
    )

    return GeminiResponse(
        conteudo=conteudo,
        model=modelo,
        input_tokens=input_tok,
        output_tokens=output_tok,
        sucesso=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Interface pública síncrona
# ─────────────────────────────────────────────────────────────────────────────

def chamar_gemini(
    prompt: str,
    system_instruction: str | None = None,
    temperatura: float | None = None,
    max_tokens: int | None = None,
    modelo: str | None = None,
    response_schema: type[BaseModel] | None = None,
) -> GeminiResponse:
    """
    Interface pública síncrona. Tenta o modelo primário com Tenacity;
    se esgotar, faz fallback automático para o modelo secundário.
    Nunca lança exceção — sempre retorna GeminiResponse.
    """
    modelo_alvo = modelo or MODELO_PRIMARIO
    temp        = temperatura if temperatura is not None else settings.GEMINI_TEMP
    max_tok     = max_tokens or settings.GEMINI_MAX_TOKENS

    try:
        return _chamar_gemini_com_retry(
            prompt, system_instruction, temp, max_tok, modelo_alvo, response_schema,
        )
    except Exception as e:
        # Tenacity esgotado → tenta fallback se estava no modelo primário
        if modelo_alvo == MODELO_PRIMARIO:
            logger.warning(
                "🔄 Tenacity esgotado para %s → tentando fallback %s",
                MODELO_PRIMARIO, MODELO_FALLBACK,
            )
            try:
                return _chamar_gemini_com_retry(
                    prompt, system_instruction, temp, max_tok,
                    MODELO_FALLBACK, response_schema,
                )
            except Exception as e_fallback:
                logger.error("❌ Fallback %s também falhou: %s", MODELO_FALLBACK, e_fallback)
                erro_msg = str(e_fallback)
        else:
            erro_msg = str(e)

        return GeminiResponse(
            conteudo="",
            model=modelo_alvo,
            sucesso=False,
            erro=erro_msg[:300],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Interface pública assíncrona (para uso em contextos async)
# ─────────────────────────────────────────────────────────────────────────────

async def chamar_gemini_async(
    prompt: str,
    system_instruction: str | None = None,
    temperatura: float | None = None,
    max_tokens: int | None = None,
    response_schema: type[BaseModel] | None = None,
) -> GeminiResponse:
    """Wrapper assíncrono: executa chamar_gemini() em thread pool."""
    return await asyncio.to_thread(
        chamar_gemini,
        prompt=prompt,
        system_instruction=system_instruction,
        temperatura=temperatura,
        max_tokens=max_tokens,
        response_schema=response_schema,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Structured Output — CORRIGIDO (aceita str ou BaseModel)
# ─────────────────────────────────────────────────────────────────────────────

def chamar_gemini_estruturado(
    prompt: str,
    response_schema: type[BaseModel] | None = None,
    schema_descricao: str | None = None,
    system_instruction: str | None = None,
    temperatura: float = 0.1,
) -> dict | None:
    """
    Chama o Gemini esperando um JSON estruturado como resposta.

    CORRIGIDO: aceita dois modos de schema para compatibilidade total:

    Modo 1 — Pydantic (preferido, usa Structured Output nativo da API):
      chamar_gemini_estruturado(prompt, response_schema=QueryRewriteSchema)
      → API garante JSON perfeito, sem regex. Nunca falha o parse.

    Modo 2 — String (compatibilidade com query_transform.py existente):
      chamar_gemini_estruturado(prompt, schema_descricao='{"query_reescrita": "str"}')
      → Injeta a descrição do schema no prompt como instrução.
      → O modelo tenta seguir, mas pode falhar o parse (menos robusto).
      → Boa prática: migrar gradualmente para Modo 1.

    Retorna o dict parseado, ou None em caso de falha.
    """
    prompt_final = prompt

    if response_schema is not None:
        # Modo 1: deixa a API garantir o formato (mais robusto)
        resposta = chamar_gemini(
            prompt=prompt_final,
            system_instruction=system_instruction,
            temperatura=temperatura,
            max_tokens=512,
            response_schema=response_schema,
        )
    elif schema_descricao is not None:
        # Modo 2: injeta instruções de formato no prompt
        instrucao = (
            f"\n\nResponda APENAS com um objeto JSON válido, sem markdown, "
            f"sem explicações, seguindo exatamente esta estrutura:\n{schema_descricao}"
        )
        prompt_final = prompt + instrucao
        resposta = chamar_gemini(
            prompt=prompt_final,
            system_instruction=system_instruction,
            temperatura=temperatura,
            max_tokens=512,
            response_schema=None,  # Sem schema nativo → depende do prompt
        )
    else:
        logger.error("chamar_gemini_estruturado: forneça response_schema ou schema_descricao")
        return None

    if not resposta.sucesso or not resposta.conteudo:
        return None

    # Parse JSON — limpa possíveis marcadores de markdown residuais
    texto = resposta.conteudo.strip()
    if texto.startswith("```"):
        texto = texto.split("```")[1]
        if texto.startswith("json"):
            texto = texto[4:]
        texto = texto.strip()

    try:
        return json.loads(texto)
    except json.JSONDecodeError as e:
        logger.error(
            "❌ JSON parse falhou: %s | conteúdo: %.100s",
            e, resposta.conteudo,
        )
        return None