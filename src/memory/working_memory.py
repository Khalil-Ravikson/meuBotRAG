"""
memory/working_memory.py — Memória de Trabalho da Conversa Atual
=================================================================

O QUE É "WORKING MEMORY" NESTE CONTEXTO?
─────────────────────────────────────────
  É o histórico da conversa ativa mais os sinais de contexto da sessão.
  Vive no Redis com TTL curto (30 min de inatividade).
  É destruída ao "voltar ao menu" ou após inatividade.

  Analogia humana:
    Long-Term Memory  = o que o aluno estudou no semestre passado
    Working Memory    = o que está na cabeça AGORA durante esta conversa

ESTRUTURAS NO REDIS:
─────────────────────
  chat:{session_id}         → List de mensagens JSON (histórico de turns)
  mem:work:{session_id}     → Hash com sinais de sessão (tool_usada, rota, etc.)

PROBLEMA QUE RESOLVEMOS — O TOKEN BUDGET:
──────────────────────────────────────────
  O sistema antigo (RedisChatMessageHistory do LangChain) armazenava tudo
  e enviava tudo para o LLM. Num bot de WhatsApp académico:
    - Uma conversa típica: 10-20 turns
    - Cada turn médio: 150 tokens (pergunta + resposta)
    - 20 turns × 150 = 3.000 tokens de histórico
    - + system prompt + contexto RAG + pergunta atual = ~5.000 tokens
    - Free tier Gemini: 1M TPM mas apenas 15 RPM

  Nossa solução em 3 camadas:
    1. Sliding Window  → máximo N turns recentes
    2. Token Budget    → se ainda ultrapassa, trunca mensagens mais antigas
    3. Compressão      → mensagens > MAX_CHARS são resumidas

  Resultado: histórico enviado ao Gemini ≤ 1.200 tokens garantido
  vs. até 3.000 tokens na versão anterior.

FORMATO DAS MENSAGENS NO REDIS:
────────────────────────────────
  [
    {"role": "user",      "content": "quando é a matrícula?",       "ts": 1700000001},
    {"role": "assistant", "content": "A matrícula de veteranos é...", "ts": 1700000002},
    ...
  ]

  role: "user" | "assistant"  (compatível com a API do Gemini diretamente)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from src.infrastructure.redis_client import get_redis_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

_TTL_CONVERSA   = 1800   # 30 min → reset automático após inatividade
_TTL_SINAIS     = 1800   # Mesmo TTL para sinais de sessão

# Janela deslizante: quantos turns (par user+assistant = 1 turn) manter
_MAX_TURNS      = 8      # 8 turns = 16 mensagens = equilibrio contexto/tokens

# Budget máximo de tokens no histórico enviado ao LLM
# Estimativa conservadora: 1 char ≈ 0.4 tokens (português)
# 1200 tokens × 2.5 chars/token ≈ 3.000 chars
_MAX_CHARS_HIST = 3_000

# Mensagens longas são truncadas neste limite (resposta do bot pode ser extensa)
_MAX_CHARS_MSG  = 400   # ~160 tokens por mensagem — suficiente para contexto

_PREFIX_CHAT = "chat:"
_PREFIX_WORK = "mem:work:"

# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

Papel = Literal["user", "assistant"]


@dataclass
class Mensagem:
    """Mensagem de histórico normalizada."""
    role:      Papel
    content:   str
    timestamp: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "ts": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict) -> "Mensagem":
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            timestamp=d.get("ts", 0),
        )

    @property
    def content_truncado(self) -> str:
        """Retorna conteúdo truncado para economizar tokens no histórico."""
        if len(self.content) <= _MAX_CHARS_MSG:
            return self.content
        return self.content[:_MAX_CHARS_MSG] + "…"


@dataclass
class HistoricoCompactado:
    """
    Resultado final do histórico pronto para envio ao Gemini.

    O Gemini aceita messages como lista de dicts:
      [{"role": "user", "parts": [{"text": "..."}]}, ...]

    Ou como string formatada (que usamos aqui por simplicidade e controlo).
    """
    mensagens:       list[dict]   # Para API do Gemini (lista de dicts)
    texto_formatado: str          # Para injeção direta no prompt como string
    total_chars:     int
    turns_incluidos: int


# ─────────────────────────────────────────────────────────────────────────────
# API principal
# ─────────────────────────────────────────────────────────────────────────────

def adicionar_mensagem(session_id: str, role: Papel, content: str) -> None:
    """
    Adiciona uma mensagem ao histórico da sessão.

    Usa RPUSH para manter ordem cronológica (mais recente no final da lista).
    Aplica ltrim imediato para não deixar o Redis crescer indefinidamente.
    """
    r = get_redis_text()
    key = f"{_PREFIX_CHAT}{session_id}"
    msg = Mensagem(role=role, content=content)

    try:
        r.rpush(key, json.dumps(msg.to_dict(), ensure_ascii=False))
        # Mantém no máximo _MAX_TURNS × 2 mensagens (user + assistant por turn)
        r.ltrim(key, -(_MAX_TURNS * 2), -1)
        r.expire(key, _TTL_CONVERSA)
    except Exception as e:
        logger.warning("⚠️  adicionar_mensagem [%s]: %s", session_id, e)


def get_historico_compactado(session_id: str) -> HistoricoCompactado:
    """
    Retorna o histórico da sessão pronto para envio ao Gemini.

    ALGORITMO DE COMPACTAÇÃO:
    ──────────────────────────
      1. Carrega as últimas _MAX_TURNS × 2 mensagens do Redis
      2. Trunca mensagens individuais muito longas (_MAX_CHARS_MSG)
      3. Aplica budget total: se somar > _MAX_CHARS_HIST, descarta
         as mensagens mais antigas (sempre mantendo pares user/assistant)
      4. Garante que sempre começa com mensagem "user" (Gemini exige)

    POR QUE MANTER PARES?
      O Gemini rejeita histórico que começa com "assistant" ou que tem
      dois "user" consecutivos. Sempre descartamos pares completos para
      manter a alternância válida.
    """
    r = get_redis_text()
    key = f"{_PREFIX_CHAT}{session_id}"

    try:
        raw = r.lrange(key, 0, -1)
        if not raw:
            return _historico_vazio()

        # Deserializa e trunca mensagens individuais
        msgs = []
        for item in raw:
            try:
                d = json.loads(item)
                m = Mensagem.from_dict(d)
                msgs.append(Mensagem(
                    role=m.role,
                    content=m.content_truncado,
                    timestamp=m.timestamp,
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        if not msgs:
            return _historico_vazio()

        # Aplica budget total de chars
        msgs = _aplicar_budget(msgs)

        # Garante início com "user"
        msgs = _garantir_inicio_user(msgs)

        if not msgs:
            return _historico_vazio()

        # Formata para API do Gemini e para string
        msgs_api = [{"role": m.role, "parts": [{"text": m.content}]} for m in msgs]
        texto = _formatar_como_string(msgs)
        total = sum(len(m.content) for m in msgs)
        turns = sum(1 for m in msgs if m.role == "user")

        return HistoricoCompactado(
            mensagens=msgs_api,
            texto_formatado=texto,
            total_chars=total,
            turns_incluidos=turns,
        )

    except Exception as e:
        logger.warning("⚠️  get_historico_compactado [%s]: %s", session_id, e)
        return _historico_vazio()


def set_sinal(session_id: str, chave: str, valor: str) -> None:
    """
    Armazena um sinal de contexto da sessão (tool usada, rota, etc.).

    SINAIS ÚTEIS:
      tool_usada:          "consultar_calendario_academico"
      rota:                "CALENDARIO"
      ultimo_topico:       "matrícula veteranos 2026.1"
      confianca_roteamento: "alta"
    """
    r = get_redis_text()
    key = f"{_PREFIX_WORK}{session_id}"
    try:
        r.hset(key, chave, valor)
        r.expire(key, _TTL_SINAIS)
    except Exception as e:
        logger.warning("⚠️  set_sinal [%s/%s]: %s", session_id, chave, e)


def get_sinais(session_id: str) -> dict[str, str]:
    """Retorna todos os sinais de contexto da sessão atual."""
    r = get_redis_text()
    key = f"{_PREFIX_WORK}{session_id}"
    try:
        return r.hgetall(key) or {}
    except Exception:
        return {}


def limpar_sessao(session_id: str) -> None:
    """
    Limpa completamente a Working Memory da sessão.
    Chamado quando utilizador digita "voltar", "oi", "reiniciar".
    """
    r = get_redis_text()
    try:
        r.delete(f"{_PREFIX_CHAT}{session_id}")
        r.delete(f"{_PREFIX_WORK}{session_id}")
        logger.debug("🗑️  Working memory limpa: %s", session_id)
    except Exception as e:
        logger.warning("⚠️  limpar_sessao [%s]: %s", session_id, e)


def get_ultimo_turn(session_id: str) -> tuple[str, str] | None:
    """
    Retorna o último par (pergunta_user, resposta_assistant).
    Usado pelo extractor de fatos para analisar o turn mais recente.
    """
    r = get_redis_text()
    key = f"{_PREFIX_CHAT}{session_id}"
    try:
        # Pega as últimas 2 mensagens
        raw = r.lrange(key, -2, -1)
        if len(raw) < 2:
            return None
        d0 = json.loads(raw[0])
        d1 = json.loads(raw[1])
        if d0.get("role") == "user" and d1.get("role") == "assistant":
            return d0.get("content", ""), d1.get("content", "")
    except Exception:
        pass
    return None


def get_ultimos_n_turns(session_id: str, n: int = 5) -> list[dict]:
    """
    Retorna os últimos N turns para a rotina de extração de fatos.
    Formato: lista de {"role": ..., "content": ...}
    """
    r = get_redis_text()
    key = f"{_PREFIX_CHAT}{session_id}"
    try:
        raw = r.lrange(key, -(n * 2), -1)
        return [json.loads(item) for item in raw if item]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Funções internas de compactação
# ─────────────────────────────────────────────────────────────────────────────

def _aplicar_budget(msgs: list[Mensagem]) -> list[Mensagem]:
    """
    Remove mensagens antigas até que o total de chars fique dentro do budget.

    ESTRATÉGIA:
      Percorre da mais antiga para a mais recente e descarta pares
      até o total estar dentro do limite.
      Sempre descarta pares completos (user + assistant) para manter
      a alternância válida exigida pelo Gemini.
    """
    total = sum(len(m.content) for m in msgs)
    if total <= _MAX_CHARS_HIST:
        return msgs

    # Trabalha com lista mutável de pares
    # Agrupa em pares: [(user, assistant), (user, assistant), ...]
    pares: list[tuple[Mensagem, Mensagem | None]] = []
    i = 0
    while i < len(msgs):
        if msgs[i].role == "user":
            if i + 1 < len(msgs) and msgs[i + 1].role == "assistant":
                pares.append((msgs[i], msgs[i + 1]))
                i += 2
            else:
                pares.append((msgs[i], None))
                i += 1
        else:
            i += 1  # Descarta assistants sem user correspondente

    # Remove pares do início até ficar dentro do budget
    while pares and total > _MAX_CHARS_HIST:
        par = pares.pop(0)
        total -= len(par[0].content)
        if par[1]:
            total -= len(par[1].content)

    # Reconstrói lista plana
    resultado: list[Mensagem] = []
    for user_msg, asst_msg in pares:
        resultado.append(user_msg)
        if asst_msg:
            resultado.append(asst_msg)

    logger.debug(
        "✂️  Budget aplicado: %d chars restantes em %d mensagens",
        sum(len(m.content) for m in resultado), len(resultado),
    )
    return resultado


def _garantir_inicio_user(msgs: list[Mensagem]) -> list[Mensagem]:
    """
    Remove mensagens do início até que comece com "user".
    O Gemini rejeita histórico que começa com "assistant".
    """
    while msgs and msgs[0].role != "user":
        msgs.pop(0)
    return msgs


def _formatar_como_string(msgs: list[Mensagem]) -> str:
    """
    Formata o histórico como string para injeção no prompt.
    Formato compacto: "Aluno: ...\nAssistente: ...\n"
    """
    linhas = []
    for m in msgs:
        prefixo = "Aluno" if m.role == "user" else "Assistente"
        linhas.append(f"{prefixo}: {m.content}")
    return "\n".join(linhas)


def _historico_vazio() -> HistoricoCompactado:
    return HistoricoCompactado(
        mensagens=[],
        texto_formatado="",
        total_chars=0,
        turns_incluidos=0,
    )