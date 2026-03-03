"""
providers/gemini_provider.py — Provedor Gemini Flash 2.0 com Tenacity (V4)
========================================================================================

ATUALIZAÇÃO COOKBOOK:
───────────────────
  - Structured Outputs: Usa Pydantic e `response_schema` nativo para garantir JSON perfeito,
    eliminando a necessidade de funções complexas de regex.
  - Prompting XML: Substitui `[BLOCOS]` por `<tags_xml>` conforme recomendado pela Google
    para melhorar o RAG e evitar alucinações.
  - Mantém o Tenacity e o backoff exponencial para resiliência.
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
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log
)

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

MODELO_PRIMARIO  = "gemini-2.0-flash"
MODELO_FALLBACK  = "gemini-1.5-flash"


# -------------------------------------------------------------------------
# SCHEMAS PYDANTIC (Cookbook: Structured Outputs)
# O Gemini garante devolver o JSON exatamente com esta estrutura.
# -------------------------------------------------------------------------
class QueryRewriteSchema(BaseModel):
    query_reescrita: str = Field(description="A pergunta reescrita com termos técnicos para busca documental")
    palavras_chave: list[str] = Field(description="Lista de palavras-chave extraídas da pergunta")

class ExtracaoFatosSchema(BaseModel):
    fatos: list[str] = Field(description="Lista de factos objetivos extraídos sobre o aluno. Vazia se não houver.")


@dataclass
class GeminiResponse:
    """Resposta normalizada do Gemini."""
    conteudo:    str
    model:       str
    input_tokens:  int = 0
    output_tokens: int = 0
    sucesso:     bool = True
    erro:        str = ""

    @property
    def tokens_total(self) -> int:
        return self.input_tokens + self.output_tokens


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Retorna cliente Gemini singleton."""
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY não configurada.")
    client = genai.Client(api_key=api_key)
    logger.info("✅ Cliente Gemini inicializado | modelo=%s", MODELO_PRIMARIO)
    return client


# -------------------------------------------------------------------------
# CONFIGURAÇÃO DO TENACITY (Resiliência)
# -------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def _chamar_gemini_com_retry(
    prompt: str,
    system_instruction: str | None = None,
    temperatura: float | None = None,
    max_tokens: int | None = None,
    modelo: str | None = None,
    response_schema: type[BaseModel] | None = None,
) -> GeminiResponse:
    """
    Função interna síncrona protegida por retry inteligente.
    Agora aceita `response_schema` para saídas JSON estruturadas.
    """
    client = get_gemini_client()
    modelo_alvo = modelo or MODELO_PRIMARIO
    temp = temperatura if temperatura is not None else settings.GEMINI_TEMP
    max_tok = max_tokens or settings.GEMINI_MAX_TOKENS

    # Constrói os parâmetros do GenerateContentConfig dinamicamente
    config_kwargs: dict[str, Any] = {
        "temperature": temp,
        "max_output_tokens": max_tok,
        "safety_settings": [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
        ],
    }

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    # Integração Nativa de Structured Outputs (JSON sem regex!)
    if response_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_kwargs)

    try:
        resposta = client.models.generate_content(
            model=modelo_alvo,
            contents=prompt,
            config=config,
        )

        usage = getattr(resposta, "usage_metadata", None)
        input_tok  = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

        conteudo = resposta.text or ""

        logger.debug(
            "✅ Gemini | modelo=%s | in=%d | out=%d",
            modelo_alvo, input_tok, output_tok,
        )

        return GeminiResponse(
            conteudo=conteudo,
            model=modelo_alvo,
            input_tokens=input_tok,
            output_tokens=output_tok,
            sucesso=True,
        )

    except Exception as e:
        err_str = str(e).lower()
        is_rate_limit = "429" in err_str or "quota" in err_str or "rate limit" in err_str
        is_overloaded  = "503" in err_str or "overloaded" in err_str

        if is_rate_limit or is_overloaded:
            logger.warning("⏳ Rate Limit do Gemini atingido. O Tenacity vai gerir o backoff...")
            raise 

        logger.error("❌ Erro fatal Gemini: %s", e)
        return GeminiResponse(conteudo="", model=modelo_alvo, sucesso=False, erro=str(e))


def chamar_gemini(
    prompt: str,
    system_instruction: str | None = None,
    temperatura: float | None = None,
    max_tokens: int | None = None,
    modelo: str | None = None,
    response_schema: type[BaseModel] | None = None,
) -> GeminiResponse:
    """Envolve a chamada protegida pelo Tenacity com try/except final para fallback."""
    try:
        return _chamar_gemini_com_retry(
            prompt, system_instruction, temperatura, max_tokens, modelo, response_schema
        )
    except Exception as e:
        if (modelo or MODELO_PRIMARIO) == MODELO_PRIMARIO:
            logger.warning("🔄 Tenacity esgotado para %s. A tentar fallback para %s...", MODELO_PRIMARIO, MODELO_FALLBACK)
            try:
                return _chamar_gemini_com_retry(
                    prompt, system_instruction, temperatura, max_tokens, MODELO_FALLBACK, response_schema
                )
            except Exception as e_fallback:
                logger.error("❌ Fallback também falhou: %s", e_fallback)
                
        return GeminiResponse(
            conteudo="", 
            model=MODELO_PRIMARIO, 
            sucesso=False, 
            erro="Max retries esgotados em todos os modelos"
        )


async def chamar_gemini_async(
    prompt: str,
    system_instruction: str | None = None,
    temperatura: float | None = None,
    max_tokens: int | None = None,
    response_schema: type[BaseModel] | None = None,
) -> GeminiResponse:
    return await asyncio.to_thread(
        chamar_gemini,
        prompt=prompt,
        system_instruction=system_instruction,
        temperatura=temperatura,
        max_tokens=max_tokens,
        response_schema=response_schema,
    )


def chamar_gemini_estruturado(
    prompt: str,
    response_schema: type[BaseModel],
    system_instruction: str | None = None,
    temperatura: float = 0.1,
) -> dict | None:
    """
    Nova Versão (Cookbook): Usa o response_schema nativo do Gemini.
    O modelo devolve sempre um JSON limpo e exato. Sem Regex!
    """
    resposta = chamar_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        temperatura=temperatura,
        max_tokens=512,
        response_schema=response_schema
    )

    if not resposta.sucesso or not resposta.conteudo:
        return None

    try:
        # A resposta é garantidamente um JSON string limpo graças à API
        return json.loads(resposta.conteudo)
    except json.JSONDecodeError:
        logger.error("❌ Falha crítica: Gemini quebrou o Structured Output: %s", resposta.conteudo)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompts Especializados (Cookbook: XML Tags)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_UEMA = """Você é o Assistente Virtual da UEMA (Universidade Estadual do Maranhão), Campus Paulo VI.
Responda sempre em português brasileiro, de forma objetiva e precisa.
Use APENAS as informações fornecidas no contexto — NUNCA invente datas, vagas ou contatos.
Se a informação não estiver no contexto, diga que não está disponível e sugira consultar uema.br.
Respostas curtas: máximo 3 parágrafos ou 6 itens. Use *negrito* para datas e termos importantes."""


def montar_prompt_geracao(
    pergunta: str,
    contexto_rag: str,
    working_memory: dict | None = None,
    fatos_usuario: list[str] | None = None,
) -> str:
    """
    Monta o prompt usando XML Tags, como recomendado no Google Gemini Cookbook.
    As tags XML ajudam o modelo a separar perfeitamente as instruções dos dados.
    """
    blocos: list[str] = []

    if fatos_usuario:
        fatos_str = "\n".join(f"- {f}" for f in fatos_usuario[:5]) 
        blocos.append(f"<perfil_aluno>\n{fatos_str}\n</perfil_aluno>")

    if working_memory:
        mem_parts = []
        if topico := working_memory.get("ultimo_topico"):
            mem_parts.append(f"Último assunto: {topico}")
        if tool := working_memory.get("tool_usada"):
            mem_parts.append(f"Área consultada: {tool}")
        if mem_parts:
            blocos.append(f"<contexto_conversa>\n" + "\n".join(mem_parts) + "\n</contexto_conversa>")

    if contexto_rag:
        blocos.append(f"<informacao_documentos>\n{contexto_rag}\n</informacao_documentos>")
    else:
        blocos.append("<informacao_documentos>\nNenhuma informação específica encontrada para esta pergunta.\n</informacao_documentos>")

    blocos.append(f"<pergunta_aluno>\n{pergunta}\n</pergunta_aluno>")

    return "\n\n".join(blocos)


PROMPT_QUERY_REWRITE = """Você reescreve perguntas de alunos para melhorar a busca em documentos académicos.

Tarefa: Reescreva a pergunta para incluir termos técnicos relevantes.

Fatos do aluno (use se relevante):
<fatos>
{fatos}
</fatos>

Pergunta original: <pergunta>{pergunta}</pergunta>

Siga os exemplos abaixo para perceber a estrutura esperada:
- "quando é minha prova?" → query_reescrita="datas provas avaliações finais 2026", palavras_chave=["prova", "avaliação", "data"]
- "como me inscrevo?" → query_reescrita="procedimento inscrição PAES 2026 documentos necessários", palavras_chave=["inscrição", "PAES", "documentos"]
"""


PROMPT_EXTRACAO_FATOS = """Analise a conversa abaixo e extraia factos objetivos sobre o aluno.

<conversa>
{conversa}
</conversa>

Extraia APENAS factos verificáveis (não suposições).
Exemplos de factos válidos:
- "Aluno do curso de Engenharia Civil"
- "Inscrito no PAES 2026 categoria BR-PPI"
- "Dúvida sobre matrícula veteranos 2026.1"
"""