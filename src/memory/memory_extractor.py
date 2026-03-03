"""
memory/memory_extractor.py — Extractor de Fatos em Background (V4 — Corrigido)
================================================================================

BUGS CORRIGIDOS vs versão anterior:
─────────────────────────────────────
  BUG 1 (CRÍTICO — schema_descricao): chamar_gemini_estruturado() era chamado
    com schema_descricao='{"fatos": ["string"]}'. Com o provider antigo isto
    gerava TypeError silencioso (argumento inexistente). Mesmo com a correção
    do provider, a abordagem de string é frágil para listas.
    CORRIGIDO: usa response_schema=ExtracaoFatosSchema (Pydantic nativo).
    A API garante que "fatos" é sempre uma lista de strings válida.

  BUG 2 (MENOR — filtro de fatos insuficiente): _validar_fatos() apenas
    verificava endswith("?") para descartar perguntas. Frases como
    "Perguntou sobre matrícula" passavam sendo meta-descrições inúteis.
    CORRIGIDO: filtros adicionais para meta-descrições e padrões inválidos.
"""
from __future__ import annotations

import logging
import time

from src.memory.long_term_memory import guardar_fatos_batch
from src.memory.working_memory import get_ultimos_n_turns, get_sinais, set_sinal
from src.providers.gemini_provider import (
    PROMPT_EXTRACAO_FATOS,
    ExtracaoFatosSchema,        # ← Schema Pydantic (corrigido)
    chamar_gemini_estruturado,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

_MIN_TURNS_PARA_EXTRACAO  = 2     # Mínimo de turns antes de tentar extração
_COOLDOWN_EXTRACAO_S      = 120   # 2 min entre extrações do mesmo utilizador
_TURNS_PARA_ANALISE       = 6     # Máximo de turns recentes a analisar
_SINAL_ULTIMA_EXTRACAO    = "ultima_extracao_ts"
_MIN_CHARS_FATO           = 15    # Fatos menores que isto são descartados

# Prefixos de meta-descrição que indicam que o Gemini descreveu a conversa
# em vez de extrair fatos sobre o aluno — padrão a descartar
_PREFIXOS_META = (
    "perguntou sobre",
    "questionou sobre",
    "demonstrou interesse",
    "demonstrou dúvida",
    "o aluno perguntou",
    "o utilizador perguntou",
    "bot respondeu",
    "assistente informou",
    "não foi possível",
    "informação não disponível",
)

# Palavras obrigatórias: um fato válido deve conter ao menos 1 destes termos
# ou ser uma afirmação direta sobre o aluno (heurística de qualidade)
_TERMOS_DE_FATO = frozenset({
    "curso", "turno", "semestre", "matrícula", "inscri",
    "categoria", "campus", "período", "trancamento", "reingresso",
    "veterano", "calouro", "engenharia", "direito", "medicina",
    "paes", "br-ppi", "br-q", "pcd", "noturno", "diurno",
    "coordenação", "departamento", "bolsa", "auxílio",
    "dificuldade", "dúvida recorrente", "frequência",
})


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def extrair_fatos_do_ultimo_turn(user_id: str, session_id: str) -> int:
    """
    Analisa os últimos turns e extrai fatos para a Long-Term Memory.
    Tolerante a falhas — erros são logados mas nunca propagados.
    """
    try:
        return _extrair_com_seguranca(user_id, session_id)
    except Exception as e:
        logger.debug("ℹ️  Extração ignorada [%s]: %s", user_id, e)
        return 0


def forcar_extracao(user_id: str, session_id: str) -> int:
    """Força extração ignorando cooldown. Para uso em debug/admin."""
    try:
        return _executar_extracao(user_id, session_id)
    except Exception as e:
        logger.warning("⚠️  Extração forçada falhou [%s]: %s", user_id, e)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Lógica interna
# ─────────────────────────────────────────────────────────────────────────────

def _extrair_com_seguranca(user_id: str, session_id: str) -> int:
    """Verifica pré-condições antes de chamar a extração."""

    # ── Verifica cooldown ─────────────────────────────────────────────────────
    sinais = get_sinais(session_id)
    try:
        ultima_ts = float(sinais.get(_SINAL_ULTIMA_EXTRACAO, "0"))
    except ValueError:
        ultima_ts = 0.0

    agora = time.time()
    restante = _COOLDOWN_EXTRACAO_S - (agora - ultima_ts)
    if restante > 0:
        logger.debug("⏳ Extração em cooldown [%s]: %.0fs restantes", user_id, restante)
        return 0

    # ── Verifica turns suficientes ────────────────────────────────────────────
    turns = get_ultimos_n_turns(session_id, n=_TURNS_PARA_ANALISE)
    n_user_turns = sum(1 for t in turns if t.get("role") == "user")

    if n_user_turns < _MIN_TURNS_PARA_EXTRACAO:
        logger.debug(
            "ℹ️  Poucos turns [%s]: %d/%d",
            user_id, n_user_turns, _MIN_TURNS_PARA_EXTRACAO,
        )
        return 0

    # ── Executa e actualiza cooldown ──────────────────────────────────────────
    guardados = _executar_extracao(user_id, session_id, turns)

    # Actualiza sempre (mesmo com 0 fatos) para evitar re-tentativa imediata
    set_sinal(session_id, _SINAL_ULTIMA_EXTRACAO, str(agora))

    return guardados


def _executar_extracao(
    user_id: str,
    session_id: str,
    turns: list[dict] | None = None,
) -> int:
    """Executa a extração de fatos via Gemini."""

    if turns is None:
        turns = get_ultimos_n_turns(session_id, n=_TURNS_PARA_ANALISE)

    if not turns:
        return 0

    conversa_formatada = _formatar_conversa(turns)
    if not conversa_formatada or len(conversa_formatada) < 50:
        return 0

    # ── Chama Gemini com Structured Output nativo ─────────────────────────────
    # CORRIGIDO: response_schema=ExtracaoFatosSchema em vez de schema_descricao
    # A API garante que "fatos" é sempre list[str], sem falhas de parse.
    # temperatura=0.05: extremamente conservador — não inventa fatos.
    prompt = PROMPT_EXTRACAO_FATOS.format(conversa=conversa_formatada)

    resultado = chamar_gemini_estruturado(
        prompt=prompt,
        response_schema=ExtracaoFatosSchema,   # ← CORRIGIDO
        temperatura=0.05,
    )

    if not resultado:
        logger.debug("ℹ️  Extração sem resultado [%s]", user_id)
        return 0

    fatos_brutos: list = resultado.get("fatos", [])
    if not fatos_brutos:
        return 0

    # ── Valida e filtra ───────────────────────────────────────────────────────
    fatos_validos = _validar_fatos(fatos_brutos)
    if not fatos_validos:
        logger.debug("ℹ️  Todos os %d fatos candidatos foram filtrados [%s]",
                     len(fatos_brutos), user_id)
        return 0

    # ── Guarda na Long-Term Memory ────────────────────────────────────────────
    guardados = guardar_fatos_batch(user_id, fatos_validos)

    if guardados:
        logger.info(
            "🧠 Fatos extraídos [%s]: %d novos / %d candidatos / %d filtrados",
            user_id, guardados, len(fatos_brutos),
            len(fatos_brutos) - len(fatos_validos),
        )

    return guardados


def _formatar_conversa(turns: list[dict]) -> str:
    """Formata turns para o prompt de extração em formato compacto."""
    linhas = []
    for turn in turns:
        role    = turn.get("role", "")
        content = turn.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            linhas.append(f"Aluno: {content[:250]}")
        elif role == "assistant":
            linhas.append(f"Bot: {content[:150]}")
    return "\n".join(linhas)


def _validar_fatos(fatos_brutos: list) -> list[str]:
    """
    Filtra fatos inválidos, genéricos ou que sejam meta-descrições da conversa.

    FILTROS APLICADOS (em ordem):
      1. Não é string → descarta
      2. Comprimento < _MIN_CHARS_FATO → muito vago ("ok", "certo")
      3. É uma pergunta (termina em "?") → não é fato, é dúvida
      4. Começa com prefixo de meta-descrição → o Gemini descreveu a
         conversa em vez de extrair um fato sobre o aluno
      5. Não contém nenhum termo de domínio UEMA → provável ruído genérico
         (ex: "O aluno demonstrou interesse" sem especificar em quê)

    NOTA: O filtro de termos (passo 5) é a heurística mais importante.
    Permite fatos como "Aluno do curso de Direito, turno noturno" mas
    bloqueia "O utilizador fez uma pergunta ao bot".
    """
    fatos_validos: list[str] = []

    for item in fatos_brutos:
        # 1. Tipo
        if not isinstance(item, str):
            continue

        fato = item.strip()

        # 2. Comprimento mínimo
        if len(fato) < _MIN_CHARS_FATO:
            logger.debug("🔍 Fato descartado (curto): %r", fato)
            continue

        # 3. É uma pergunta
        if fato.endswith("?"):
            logger.debug("🔍 Fato descartado (pergunta): %r", fato)
            continue

        # 4. Meta-descrição da conversa
        fato_lower = fato.lower()
        if any(fato_lower.startswith(p) for p in _PREFIXOS_META):
            logger.debug("🔍 Fato descartado (meta-descrição): %r", fato)
            continue

        # 5. Deve conter pelo menos 1 termo de domínio relevante
        if not any(termo in fato_lower for termo in _TERMOS_DE_FATO):
            logger.debug("🔍 Fato descartado (sem termo de domínio): %r", fato)
            continue

        fatos_validos.append(fato)

    return fatos_validos