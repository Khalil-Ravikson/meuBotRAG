"""
memory/redis_memory.py — Histórico de conversa + estado do menu
===============================================================
Substitui redis_history.py e a parte Redis do menu_service.py.

FUNCIONALIDADES PRESERVADAS DO redis_history.py:
  - Sanitização de tool_calls órfãos (causa do erro 400 do Groq)
    Quando o Groq retorna erro 400 (tool_use_failed), o LangChain já salvou
    a AIMessage com tool_calls no Redis, mas o ToolMessage de resposta nunca
    chegou. Na próxima chamada, o histórico fica inválido e o Groq falha de novo.
    A função _sanitizar_mensagens() remove esses pares incompletos.

  - Sliding window de 20 mensagens (corte sempre em HumanMessage)
  - Truncamento via clear() + add_messages() (evita NotImplementedError)

ADICIONADO:
  - Estado do menu por usuário (migrado de menu_service.py)
  - Contexto persistente do usuário (nome, curso, última intenção)
  - Usa redis_client singleton (sem instâncias espalhadas)
"""
from __future__ import annotations
import json
import logging

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage

from src.infrastructure.redis_client import get_redis
from src.infrastructure.settings import settings
from src.domain.entities import EstadoMenu

logger = logging.getLogger(__name__)

_TTL_HISTORICO = 1800   # 30min de inatividade reseta histórico
_TTL_ESTADO    = 1800   # 30min de inatividade reseta estado do menu
_TTL_CONTEXTO  = 3600   # contexto do usuário dura 1h
_SLIDING_WINDOW = 20    # máximo de mensagens na janela deslizante


# =============================================================================
# Sanitização de tool_calls órfãos (preservado de redis_history.py)
# =============================================================================

def _sanitizar_mensagens(mensagens: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    """
    Remove AIMessages com tool_calls sem ToolMessage correspondente.

    Por que isso acontece?
    Quando o Groq retorna erro 400 (tool_use_failed), o LangChain já salvou
    a AIMessage com tool_calls no Redis, mas o ToolMessage de resposta nunca
    chegou. Na próxima chamada, o Groq recebe um histórico inválido e falha
    de novo.

    Retorna: (lista_limpa, quantidade_removida)
    """
    limpas: list[BaseMessage] = []
    removidas = 0
    i = 0

    while i < len(mensagens):
        msg = mensagens[i]

        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            ids_esperados = {tc["id"] for tc in msg.tool_calls if "id" in tc}

            if ids_esperados:
                proximas = mensagens[i + 1: i + 1 + len(ids_esperados)]
                ids_respondidos = {
                    m.tool_call_id
                    for m in proximas
                    if isinstance(m, ToolMessage) and hasattr(m, "tool_call_id")
                }

                if not ids_esperados.issubset(ids_respondidos):
                    # Tool call órfã — descarta AIMessage e ToolMessages parciais
                    removidas += 1
                    i += 1
                    while i < len(mensagens) and isinstance(mensagens[i], ToolMessage):
                        removidas += 1
                        i += 1
                    continue

        limpas.append(msg)
        i += 1

    return limpas, removidas


# =============================================================================
# Histórico LangChain (para RunnableWithMessageHistory)
# =============================================================================

def get_historico(session_id: str) -> RedisChatMessageHistory:
    """Retorna histórico Redis bruto para uma sessão."""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=_TTL_HISTORICO,
    )


def get_historico_limitado(session_id: str) -> RedisChatMessageHistory:
    """
    Retorna histórico sanitizado e truncado para uso no AgentCore.

    Ordem das operações (importa!):
      1. Conecta ao Redis
      2. Sanitiza tool_calls órfãos  ← corrige erro 400 do Groq
      3. Aplica sliding window de _SLIDING_WINDOW mensagens
         (corte sempre em HumanMessage, nunca no meio de um par tool)
      4. Aplica limite de MAX_HISTORY_MESSAGES do settings
      5. Persiste o histórico limpo se houve mudanças
    """
    historico = get_historico(session_id)

    try:
        msgs = historico.messages

        if not msgs:
            return historico

        # ── Passo 1: Sanitiza tool_calls órfãos ──────────────────────────────
        msgs_limpas, n_removidas = _sanitizar_mensagens(msgs)
        if n_removidas > 0:
            logger.info(
                "🧹 [%s] %d msg(s) corrompida(s) removida(s) do histórico.",
                session_id, n_removidas,
            )

        # ── Passo 2: Sliding window ───────────────────────────────────────────
        # Nunca corta no meio de um par AIMessage+ToolMessage
        if len(msgs_limpas) > _SLIDING_WINDOW:
            candidatas = msgs_limpas[-_SLIDING_WINDOW:]
            # Garante que começa em HumanMessage
            inicio_seguro = 0
            for j, m in enumerate(candidatas):
                if isinstance(m, HumanMessage):
                    inicio_seguro = j
                    break
            msgs_limpas = candidatas[inicio_seguro:]

        # ── Passo 3: Limite de MAX_HISTORY_MESSAGES ───────────────────────────
        limite = settings.MAX_HISTORY_MESSAGES
        if len(msgs_limpas) > limite:
            msgs_limpas = msgs_limpas[-limite:]
            # Garante início em HumanMessage após truncamento
            for j, m in enumerate(msgs_limpas):
                if isinstance(m, HumanMessage):
                    msgs_limpas = msgs_limpas[j:]
                    break

        # ── Passo 4: Persiste apenas se algo mudou ────────────────────────────
        if len(msgs_limpas) != len(msgs):
            historico.clear()
            historico.add_messages(msgs_limpas)
            logger.debug(
                "✂️  Histórico [%s] truncado: %d → %d msgs.",
                session_id, len(msgs), len(msgs_limpas),
            )

    except Exception as e:
        logger.warning("⚠️  Erro ao processar histórico [%s]: %s", session_id, e)

    return historico


def limpar_historico(session_id: str) -> bool:
    """
    Apaga todo o histórico de uma sessão.
    Use no comando 'reiniciar' ou quando tool_use_failed ocorrer.
    Retorna True se limpou com sucesso.
    """
    try:
        get_historico(session_id).clear()
        logger.debug("🗑️  Histórico [%s] limpo.", session_id)
        return True
    except Exception as e:
        logger.warning("⚠️  Falha ao limpar histórico [%s]: %s", session_id, e)
        return False


# =============================================================================
# Estado do menu (migrado de menu_service.py)
# =============================================================================

def get_estado_menu(user_id: str) -> EstadoMenu:
    try:
        val = get_redis().get(f"menu_state:{user_id}")
        if val:
            return EstadoMenu(val)
    except Exception:
        pass
    return EstadoMenu.MAIN


def set_estado_menu(user_id: str, estado: EstadoMenu) -> None:
    try:
        get_redis().setex(f"menu_state:{user_id}", _TTL_ESTADO, estado.value)
    except Exception as e:
        logger.warning("⚠️  Falha ao salvar estado menu [%s]: %s", user_id, e)


def clear_estado_menu(user_id: str) -> None:
    try:
        get_redis().delete(f"menu_state:{user_id}")
    except Exception:
        pass


# =============================================================================
# Contexto persistente do usuário (migrado de menu_service.py)
# =============================================================================

def get_contexto(user_id: str) -> dict:
    try:
        raw = get_redis().get(f"user_ctx:{user_id}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {}


def set_contexto(user_id: str, dados: dict) -> None:
    """Merge: atualiza campos sem sobrescrever os existentes."""
    try:
        ctx = get_contexto(user_id)
        ctx.update(dados)
        get_redis().setex(
            f"user_ctx:{user_id}", _TTL_CONTEXTO,
            json.dumps(ctx, ensure_ascii=False),
        )
    except Exception as e:
        logger.warning("⚠️  Falha ao salvar contexto [%s]: %s", user_id, e)


def clear_tudo(user_id: str) -> None:
    """Limpa histórico + estado do menu + contexto de um usuário."""
    limpar_historico(user_id)
    clear_estado_menu(user_id)
    try:
        get_redis().delete(f"user_ctx:{user_id}")
    except Exception:
        pass