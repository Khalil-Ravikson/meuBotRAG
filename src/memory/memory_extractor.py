"""
memory/memory_extractor.py — Extractor de Fatos em Background
==============================================================

INSPIRAÇÃO: Agent Memory Server (Redis)
─────────────────────────────────────────
  O repositório redis/agent-memory-server define 3 camadas de memória:
    1. Working Memory    → conversa atual (já temos em working_memory.py)
    2. Episodic Memory   → o que aconteceu em sessões passadas
    3. Semantic Memory   → fatos factuais sobre o utilizador

  Este ficheiro implementa o mecanismo de CONSOLIDAÇÃO:
  o processo que lê a Working Memory e cristaliza fatos na Semantic Memory.

O QUE É EXTRAÇÃO DE FATOS:
────────────────────────────
  É a capacidade do sistema de aprender sobre o utilizador ao longo
  do tempo sem ele ter de se repetir.

  Sessão 1 (Janeiro):
    Aluno: "Sou do curso de Engenharia Civil, noturno"
    → Extrator detecta fato: "Aluno de Engenharia Civil, turno noturno"
    → Guarda na long_term_memory

  Sessão 2 (Fevereiro):
    Aluno: "quando é minha matrícula?"
    → long_term_memory devolve: "Aluno de Engenharia Civil, turno noturno"
    → Query transform: "matrícula veteranos Engenharia Civil noturno 2026.1"
    → Busca encontra o chunk exacto → sem alucinação ✓

  Sessão 3 (Março):
    Aluno: "posso trancar?"
    → long_term_memory: "Aluno EC noturno" + "já fez matrícula 2026.1"
    → Context: sabe que é veterano, sabe o semestre → resposta precisa

QUANDO É EXECUTADO:
─────────────────────
  O extractor é chamado NO FINAL de cada turn (não bloqueia a resposta):

    AgentCore._lancar_extracao_background()
      → asyncio.ensure_future(asyncio.to_thread(extrair_fatos_do_ultimo_turn))

  Ou seja, a resposta já foi enviada ao utilizador antes de a extração começar.

  Alternativa para produção: tarefa agendada (APScheduler, Celery, cron)
  que processa fila de sessões a cada 5 minutos. Mas para este hardware
  (16GB RAM), a extração inline em background é mais simples e eficiente.

HEURÍSTICAS ANTI-RUÍDO:
────────────────────────
  Nem tudo numa conversa é um "fato útil":
  ✗ "oi tudo bem" → não é fato
  ✗ "obrigado" → não é fato
  ✗ "ok" → não é fato
  ✓ "sou do curso de direito" → fato estrutural
  ✓ "me inscrevi no PAES via BR-PPI" → fato específico
  ✓ "tenho dificuldades com as datas de matrícula" → padrão de dúvida

  O Gemini faz a filtragem com temperatura=0.05 (muito conservador)
  e o prompt instrui explicitamente a retornar [] se não houver fatos.

CUSTO:
───────
  Chamada Gemini para extração: ~200 tokens (150 input + 50 output)
  Frequência: 1 por turn (mas só quando há conteúdo suficiente)
  Com _MIN_TURNS_PARA_EXTRACAO=2 e _COOLDOWN_EXTRACAO_S=120:
  → ~1 extração a cada 2-3 minutos por utilizador activo
  → Impacto negligível no free tier (15 RPM, 1M TPM)
"""
from __future__ import annotations

import logging
import time

from src.memory.long_term_memory import guardar_fatos_batch
from src.memory.working_memory import get_ultimos_n_turns, get_sinais, set_sinal
from src.providers.gemini_provider import (
    PROMPT_EXTRACAO_FATOS,
    chamar_gemini_estruturado,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

# Mínimo de turns na sessão antes de tentar extração
# (evita extrair de conversas de 1 mensagem)
_MIN_TURNS_PARA_EXTRACAO = 2

# Cooldown entre extrações para o mesmo utilizador (em segundos)
# Evita chamar o Gemini em cada turn de conversas longas
_COOLDOWN_EXTRACAO_S = 120   # 2 minutos

# Máximo de turns recentes a passar ao Gemini para extração
_TURNS_PARA_ANALISE = 6

# Chave do sinal de cooldown na working memory
_SINAL_ULTIMA_EXTRACAO = "ultima_extracao_ts"

# Comprimento mínimo de um fato para ser considerado válido
_MIN_CHARS_FATO = 15


# ─────────────────────────────────────────────────────────────────────────────
# API principal
# ─────────────────────────────────────────────────────────────────────────────

def extrair_fatos_do_ultimo_turn(user_id: str, session_id: str) -> int:
    """
    Analisa os últimos turns da sessão e extrai fatos para a Long-Term Memory.

    Esta função é chamada em background pelo AgentCore após cada resposta.
    É tolerante a falhas — erros são logados mas não propagados.

    Retorna o número de novos fatos guardados (0 se nada extraído).
    """
    try:
        return _extrair_com_seguranca(user_id, session_id)
    except Exception as e:
        logger.debug("ℹ️  Extração de fatos ignorada [%s]: %s", user_id, e)
        return 0


def forcar_extracao(user_id: str, session_id: str) -> int:
    """
    Força a extração ignorando o cooldown.
    Usado ao fim de uma sessão longa ou por comando administrativo.
    """
    try:
        return _executar_extracao(user_id, session_id)
    except Exception as e:
        logger.warning("⚠️  Extração forçada falhou [%s]: %s", user_id, e)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Lógica interna
# ─────────────────────────────────────────────────────────────────────────────

def _extrair_com_seguranca(user_id: str, session_id: str) -> int:
    """
    Wrapper com todas as verificações de pré-condição.
    """
    # ── Verifica cooldown ─────────────────────────────────────────────────────
    sinais = get_sinais(session_id)
    ultima_ts_str = sinais.get(_SINAL_ULTIMA_EXTRACAO, "0")

    try:
        ultima_ts = float(ultima_ts_str)
    except ValueError:
        ultima_ts = 0.0

    agora = time.time()
    if agora - ultima_ts < _COOLDOWN_EXTRACAO_S:
        logger.debug(
            "⏳ Extração em cooldown [%s]: %.0fs restantes",
            user_id, _COOLDOWN_EXTRACAO_S - (agora - ultima_ts),
        )
        return 0

    # ── Verifica se há turns suficientes ─────────────────────────────────────
    turns = get_ultimos_n_turns(session_id, n=_TURNS_PARA_ANALISE)
    n_user_turns = sum(1 for t in turns if t.get("role") == "user")

    if n_user_turns < _MIN_TURNS_PARA_EXTRACAO:
        logger.debug(
            "ℹ️  Poucos turns para extração [%s]: %d/%d",
            user_id, n_user_turns, _MIN_TURNS_PARA_EXTRACAO,
        )
        return 0

    # ── Executa extração ──────────────────────────────────────────────────────
    guardados = _executar_extracao(user_id, session_id, turns)

    # Actualiza timestamp de cooldown (mesmo que 0 fatos → evita re-tentativa)
    set_sinal(session_id, _SINAL_ULTIMA_EXTRACAO, str(agora))

    return guardados


def _executar_extracao(
    user_id: str,
    session_id: str,
    turns: list[dict] | None = None,
) -> int:
    """
    Executa a extração de fatos propriamente dita.
    """
    if turns is None:
        turns = get_ultimos_n_turns(session_id, n=_TURNS_PARA_ANALISE)

    if not turns:
        return 0

    # ── Formata a conversa para o Gemini ─────────────────────────────────────
    conversa_formatada = _formatar_conversa(turns)

    if not conversa_formatada or len(conversa_formatada) < 50:
        return 0

    # ── Chama Gemini para extração ────────────────────────────────────────────
    prompt = PROMPT_EXTRACAO_FATOS.format(conversa=conversa_formatada)

    resultado = chamar_gemini_estruturado(
        prompt=prompt,
        schema_descricao='{"fatos": ["string"]}',
        temperatura=0.05,   # Muito conservador — não inventa fatos
    )

    if not resultado:
        logger.debug("ℹ️  Extração sem resultado para [%s]", user_id)
        return 0

    fatos_brutos: list[str] = resultado.get("fatos", [])
    if not fatos_brutos:
        return 0

    # ── Valida e filtra fatos ─────────────────────────────────────────────────
    fatos_validos = _validar_fatos(fatos_brutos)

    if not fatos_validos:
        return 0

    # ── Guarda na Long-Term Memory ────────────────────────────────────────────
    guardados = guardar_fatos_batch(user_id, fatos_validos)

    if guardados:
        logger.info(
            "🧠 Fatos extraídos [%s]: %d novos de %d candidatos",
            user_id, guardados, len(fatos_validos),
        )

    return guardados


def _formatar_conversa(turns: list[dict]) -> str:
    """
    Formata os turns para o prompt de extração.
    Usa formato compacto para economizar tokens.
    """
    linhas = []
    for turn in turns:
        role    = turn.get("role", "")
        content = turn.get("content", "").strip()

        if not content:
            continue

        if role == "user":
            linhas.append(f"Aluno: {content[:200]}")   # Limita por turn
        elif role == "assistant":
            # Inclui resposta do assistente para contexto, mas truncada
            linhas.append(f"Bot: {content[:150]}")

    return "\n".join(linhas)


def _validar_fatos(fatos_brutos: list) -> list[str]:
    """
    Filtra fatos inválidos, muito curtos ou genéricos.

    FILTROS APLICADOS:
      1. Não é string → descarta
      2. Menos de _MIN_CHARS_FATO chars → muito vago
      3. É uma saudação ou resposta genérica → descarta
      4. É uma pergunta (termina em "?") → não é fato, é dúvida
    """
    # Padrões que indicam não-fatos
    _NAO_FATOS = frozenset({
        "oi", "olá", "obrigado", "obrigada", "tchau", "até logo",
        "ok", "certo", "entendi", "sim", "não", "tá", "ta",
        "valeu", "vlw", "blz", "boa", "bom dia", "boa tarde",
    })

    fatos_validos = []
    for fato in fatos_brutos:
        if not isinstance(fato, str):
            continue

        fato_limpo = fato.strip()

        # Muito curto
        if len(fato_limpo) < _MIN_CHARS_FATO:
            continue

        # Saudação ou resposta genérica
        if fato_limpo.lower() in _NAO_FATOS:
            continue

        # É uma pergunta, não um fato
        if fato_limpo.endswith("?"):
            continue

        # Fato muito genérico (menos de 3 palavras = provavelmente inútil)
        palavras = fato_limpo.split()
        if len(palavras) < 3:
            continue

        fatos_validos.append(fato_limpo)

    return fatos_validos


# ─────────────────────────────────────────────────────────────────────────────
# Função de teste / diagnóstico
# ─────────────────────────────────────────────────────────────────────────────

def testar_extracao(conversa_exemplo: str) -> list[str]:
    """
    Testa o extractor com uma conversa de exemplo.
    Útil no debug/Chainlit para validar o prompts.

    Uso:
      from src.memory.memory_extractor import testar_extracao
      fatos = testar_extracao(
          "Aluno: sou do curso de direito, período noturno\\n"
          "Bot: Entendido! Posso ajudar com informações do curso.\\n"
          "Aluno: quero me inscrever pelo PAES como cotista BR-PPI"
      )
      print(fatos)
      # → ["Aluno do curso de Direito, turno noturno",
      #    "Interessado em inscrição PAES 2026 categoria BR-PPI"]
    """
    prompt = PROMPT_EXTRACAO_FATOS.format(conversa=conversa_exemplo)
    resultado = chamar_gemini_estruturado(
        prompt=prompt,
        schema_descricao='{"fatos": ["string"]}',
        temperatura=0.05,
    )
    if not resultado:
        return []
    fatos_brutos = resultado.get("fatos", [])
    return _validar_fatos(fatos_brutos)