"""
providers/gemini_provider.py — Provedor Gemini Flash 2.0 (Google AI Studio Free Tier)
========================================================================================

POR QUE MIGRAMOS DO GROQ PARA O GEMINI:
─────────────────────────────────────────
  Problema Groq:
    - Rate limits agressivos no free tier (6000 TPM, 30 RPM)
    - Custos ao escalar além do free tier
    - Modelo llama-3.1-8b-instant, embora rápido, alucina mais em datas/siglas

  Vantagem Gemini 1.5 Flash (Google AI Studio Free Tier, Fev 2026):
    - 15 RPM, 1.000.000 TPM (tokens por minuto!) — muito mais folgado
    - 1500 req/dia → suficiente para um bot académico
    - Contexto 1M tokens (útil para passar muito contexto de PDFs)
    - Custo: $0 (Free Tier do Google AI Studio)
    - Gemini 2.0 Flash: ainda mais rápido, mesmo preço ($0)

  ⚠️  ATENÇÃO: O free tier NÃO suporta fine-tuning nem Vertex AI.
       Use Google AI Studio (https://aistudio.google.com) para obter API_KEY.

BIBLIOTECA USADA:
─────────────────
  google-genai (nova, oficial do Google, substitui google-generativeai)
  pip install google-genai

  API: https://ai.google.dev/api/python/google/genai

ESTRATÉGIA DE TOKENS (ECONOMIA):
──────────────────────────────────
  ANTES (Groq + LangChain AgentExecutor):
    - System prompt: ~500 tokens
    - Histórico completo: até 2000 tokens
    - Tool definitions JSON: ~800 tokens (3 tools)
    - Contexto RAG: ~1000 tokens
    - TOTAL por request: ~4300 tokens de entrada

  AGORA (Gemini direto + pipeline manual):
    - System prompt minimalista: ~200 tokens
    - Working memory + fatos: ~150 tokens
    - Contexto RAG filtrado: ~600 tokens (híbrido encontra mais relevante)
    - Sem tool definitions (roteamento feito localmente via Redis)
    - TOTAL por request: ~950 tokens de entrada
    - ECONOMIA: ~78% menos tokens por chamada

USO DO MÓDULO:
──────────────
  from src.providers.gemini_provider import get_gemini, chamar_gemini, chamar_gemini_estruturado

  # Geração simples
  resposta = await chamar_gemini(prompt, contexto="...")

  # Com estrutura JSON garantida (para query rewriting, extração de fatos)
  dados = await chamar_gemini_estruturado(prompt, schema={"tipo": str, "query": str})
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import google.genai as genai
from google.genai import types

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração do modelo
# ─────────────────────────────────────────────────────────────────────────────

# gemini-2.0-flash → mais rápido e moderno, mantém free tier
# gemini-1.5-flash → fallback se 2.0 não estiver disponível na região
MODELO_PRIMARIO  = "gemini-2.0-flash"
MODELO_FALLBACK  = "gemini-1.5-flash"

# Limites do free tier (conservadores para evitar 429)
# 15 RPM → no máximo 1 request a cada 4 segundos com margem
_MIN_INTERVALO_S = 4.0

# Retry com backoff exponencial
_MAX_RETRIES  = 3
_BACKOFF_BASE = 5.0   # segundos — backoff agressivo para free tier

# Timestamp da última chamada (para rate limiting manual)
_ultima_chamada: float = 0.0


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


# ─────────────────────────────────────────────────────────────────────────────
# Cliente singleton
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """
    Retorna cliente Gemini singleton.

    NOTA SOBRE A API KEY:
      Obtenha em: https://aistudio.google.com/app/apikey
      Adicione ao .env: GEMINI_API_KEY=AIza...
      NÃO use Vertex AI — é pago. Use Google AI Studio.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY não configurada. "
            "Obtenha em: https://aistudio.google.com/app/apikey"
        )
    client = genai.Client(api_key=api_key)
    logger.info("✅ Cliente Gemini inicializado | modelo=%s", MODELO_PRIMARIO)
    return client


def _aguardar_rate_limit() -> None:
    """
    Controlo manual de rate limit para o free tier.

    POR QUE MANUAL E NÃO VIA RETRY?
      O free tier do Gemini retorna 429 sem Retry-After header.
      Controlar antes da chamada é mais eficiente que reagir depois.
    """
    global _ultima_chamada
    agora = time.monotonic()
    decorrido = agora - _ultima_chamada
    if decorrido < _MIN_INTERVALO_S:
        espera = _MIN_INTERVALO_S - decorrido
        logger.debug("⏳ Rate limit preventivo: aguardando %.1fs", espera)
        time.sleep(espera)
    _ultima_chamada = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# Funções de chamada
# ─────────────────────────────────────────────────────────────────────────────

def chamar_gemini(
    prompt: str,
    system_instruction: str | None = None,
    temperatura: float | None = None,
    max_tokens: int | None = None,
    modelo: str | None = None,
) -> GeminiResponse:
    """
    Chama o Gemini de forma síncrona com retry automático.

    POR QUE SÍNCRONO?
      A maior parte do código LangChain existente é síncrono.
      Manter síncrono facilita a migração incremental.
      Para uso assíncrono, use chamar_gemini_async().

    Parâmetros:
      prompt:             Mensagem principal (humana)
      system_instruction: Instrução de sistema (opcional, substitui "system prompt")
      temperatura:        0.0-1.0 (padrão: settings.GEMINI_TEMP)
      max_tokens:         Limite de tokens na resposta
      modelo:             Override do modelo (padrão: MODELO_PRIMARIO)
    """
    client = get_gemini_client()
    modelo_alvo = modelo or MODELO_PRIMARIO
    temp = temperatura if temperatura is not None else settings.GEMINI_TEMP
    max_tok = max_tokens or settings.GEMINI_MAX_TOKENS

    config = types.GenerateContentConfig(
        temperature=temp,
        max_output_tokens=max_tok,
        system_instruction=system_instruction,
        # safety_settings reduzidos para contexto académico
        # (evita bloqueios em perguntas sobre "avaliações" ou "provas")
        safety_settings=[
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
    )

    for tentativa in range(1, _MAX_RETRIES + 1):
        try:
            _aguardar_rate_limit()

            resposta = client.models.generate_content(
                model=modelo_alvo,
                contents=prompt,
                config=config,
            )

            # Extrai métricas de uso de tokens
            usage = getattr(resposta, "usage_metadata", None)
            input_tok  = getattr(usage, "prompt_token_count", 0) if usage else 0
            output_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

            conteudo = resposta.text or ""

            logger.debug(
                "✅ Gemini | modelo=%s | in=%d | out=%d | tentativa=%d",
                modelo_alvo, input_tok, output_tok, tentativa,
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
            is_rate_limit = "429" in err_str or "quota" in err_str or "rate" in err_str
            is_overloaded  = "503" in err_str or "overloaded" in err_str

            if (is_rate_limit or is_overloaded) and tentativa < _MAX_RETRIES:
                espera = _BACKOFF_BASE * (2 ** (tentativa - 1))
                logger.warning(
                    "⏳ Rate limit/Sobrecarga Gemini (tentativa %d/%d). Aguardando %.0fs...",
                    tentativa, _MAX_RETRIES, espera,
                )
                time.sleep(espera)
                continue

            # Tenta fallback para modelo anterior
            if modelo_alvo == MODELO_PRIMARIO and tentativa == _MAX_RETRIES:
                logger.warning("🔄 Tentando fallback para %s", MODELO_FALLBACK)
                modelo_alvo = MODELO_FALLBACK
                tentativa = 0
                continue

            logger.error("❌ Erro Gemini (tentativa %d/%d): %s", tentativa, _MAX_RETRIES, e)
            return GeminiResponse(
                conteudo="",
                model=modelo_alvo,
                sucesso=False,
                erro=str(e),
            )

    return GeminiResponse(conteudo="", model=modelo_alvo, sucesso=False, erro="Max retries")


async def chamar_gemini_async(
    prompt: str,
    system_instruction: str | None = None,
    temperatura: float | None = None,
    max_tokens: int | None = None,
) -> GeminiResponse:
    """
    Versão assíncrona do chamar_gemini.
    Usa asyncio.to_thread para não bloquear o event loop do FastAPI.
    """
    return await asyncio.to_thread(
        chamar_gemini,
        prompt=prompt,
        system_instruction=system_instruction,
        temperatura=temperatura,
        max_tokens=max_tokens,
    )


def chamar_gemini_estruturado(
    prompt: str,
    schema_descricao: str,
    system_instruction: str | None = None,
    temperatura: float = 0.1,
) -> dict | None:
    """
    Chama o Gemini esperando retorno em JSON estruturado.

    USADO PARA:
      1. Query Transformation: reescrever a pergunta com contexto factual
         Input:  "onde fica minha prova?"
         Output: {"query_reescrita": "local de prova Engenharia Civil UEMA 2026.1",
                  "intencao": "calendario", "sub_intencao": "local_prova"}

      2. Extração de fatos (rotina noturna):
         Input:  conversa completa
         Output: {"fatos": ["Aluno de Engenharia Civil", "Inscrito via BR-PPI"]}

    COMO GARANTIMOS JSON:
      - Pedimos explicitamente JSON no prompt
      - Baixa temperatura (0.1) reduz criatividade estrutural
      - Parsing com fallback para extrair JSON mesmo com texto envolvente
      - Retorna None se JSON inválido (chamador lida com None)

    Parâmetros:
      prompt:            Texto da tarefa
      schema_descricao:  Descrição do JSON esperado (instrui o modelo)
      temperatura:       Baixa para JSON (0.1 padrão)
    """
    instrucao_json = (
        f"{system_instruction or ''}\n\n"
        f"IMPORTANTE: Responda APENAS com JSON válido. Sem texto antes ou depois. "
        f"Schema esperado:\n{schema_descricao}"
    ).strip()

    resposta = chamar_gemini(
        prompt=prompt,
        system_instruction=instrucao_json,
        temperatura=temperatura,
        max_tokens=512,    # JSON estruturado não precisa de muito espaço
    )

    if not resposta.sucesso or not resposta.conteudo:
        return None

    # Tenta extrair JSON mesmo se o modelo adicionou texto extra
    conteudo = resposta.conteudo.strip()
    parsed = _extrair_json(conteudo)

    if parsed is None:
        logger.warning("⚠️  Gemini retornou JSON inválido: %.100s", conteudo)

    return parsed


def _extrair_json(texto: str) -> dict | None:
    """
    Tenta múltiplas estratégias para extrair JSON do texto.

    Estratégias (por ordem de confiança):
      1. Parse direto → texto já é JSON válido
      2. Extrai bloco ```json ... ``` → modelo adicionou markdown
      3. Extrai primeiro { ... } → modelo adicionou prefácio/sufácio
    """
    texto = texto.strip()

    # Estratégia 1: parse direto
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    # Estratégia 2: bloco markdown
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", texto)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Estratégia 3: primeiro JSON object
    match = re.search(r"\{[\s\S]+\}", texto)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompts especializados (única fonte da verdade dos prompts Gemini)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_UEMA = """Você é o Assistente Virtual da UEMA (Universidade Estadual do Maranhão), Campus Paulo VI.
Responda sempre em português brasileiro, de forma objetiva e precisa.
Use APENAS as informações fornecidas no contexto — NUNCA invente datas, vagas ou contatos.
Se a informação não estiver no contexto, diga que não está disponível e sugira uema.br.
Respostas curtas: máximo 3 parágrafos ou 6 itens. Use *negrito* para datas e termos importantes."""


def montar_prompt_geracao(
    pergunta: str,
    contexto_rag: str,
    working_memory: dict | None = None,
    fatos_usuario: list[str] | None = None,
) -> str:
    """
    Monta o prompt final para geração da resposta.

    ESTRUTURA DO PROMPT (otimizado para economia de tokens):
    ─────────────────────────────────────────────────────────
      [FATOS DO ALUNO]         ← apenas se existirem (Long-Term Memory)
      [CONTEXTO DA CONVERSA]   ← apenas se relevante (Working Memory)
      [INFORMAÇÕES ENCONTRADAS] ← resultado da busca híbrida
      [PERGUNTA]               ← pergunta original ou reescrita

    POR QUE ESTA ORDEM?
      O Gemini tem "recency bias" — o que aparece por último influencia mais.
      Colocamos a PERGUNTA por último para que seja o foco principal.
      O CONTEXTO RAG logo antes ancora a resposta nos dados reais.
    """
    blocos: list[str] = []

    # Fatos do utilizador (Long-Term Memory) — reduz perguntas de esclarecimento
    if fatos_usuario:
        fatos_str = "\n".join(f"- {f}" for f in fatos_usuario[:5])  # Máximo 5 fatos
        blocos.append(f"[PERFIL DO ALUNO]\n{fatos_str}")

    # Working Memory — contexto da conversa atual
    if working_memory:
        mem_parts = []
        if topico := working_memory.get("ultimo_topico"):
            mem_parts.append(f"Último assunto: {topico}")
        if tool := working_memory.get("tool_usada"):
            mem_parts.append(f"Área consultada: {tool}")
        if mem_parts:
            blocos.append(f"[CONTEXTO DA CONVERSA]\n" + "\n".join(mem_parts))

    # Contexto RAG (resultado da busca híbrida)
    if contexto_rag:
        blocos.append(f"[INFORMAÇÕES ENCONTRADAS NOS DOCUMENTOS]\n{contexto_rag}")
    else:
        blocos.append("[INFORMAÇÕES]\nNenhuma informação específica encontrada para esta pergunta.")

    # Pergunta
    blocos.append(f"[PERGUNTA DO ALUNO]\n{pergunta}")

    return "\n\n".join(blocos)


PROMPT_QUERY_REWRITE = """Você reescreve perguntas de alunos para melhorar a busca em documentos académicos.

Tarefa: Reescreva a pergunta para incluir termos técnicos relevantes.

Fatos do aluno (use se relevante):
{fatos}

Pergunta original: {pergunta}

Responda APENAS com JSON:
{{"query_reescrita": "versão expandida com termos técnicos", "palavras_chave": ["termo1", "termo2"]}}

Exemplos:
- "quando é minha prova?" → {{"query_reescrita": "datas provas avaliações finais 2026", "palavras_chave": ["prova", "avaliação", "data"]}}
- "como me inscrevo?" → {{"query_reescrita": "procedimento inscrição PAES 2026 documentos necessários", "palavras_chave": ["inscrição", "PAES", "documentos"]}}
"""


PROMPT_EXTRACAO_FATOS = """Analise a conversa abaixo e extraia fatos objetivos sobre o aluno.

Conversa:
{conversa}

Extraia APENAS fatos verificáveis (não suposições).
Exemplos de fatos válidos:
- "Aluno do curso de Engenharia Civil"
- "Inscrito no PAES 2026 categoria BR-PPI"
- "Dúvida sobre matrícula veteranos 2026.1"

Responda APENAS com JSON:
{{"fatos": ["fato1", "fato2", "fato3"]}}

Se não houver fatos claros, retorne: {{"fatos": []}}
"""