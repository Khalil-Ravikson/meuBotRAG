"""
redis_history.py

Gerencia o histórico de conversas no Redis com:
- Sliding window (janela deslizante) de 20 mensagens
- Sanitização de tool_calls órfãos (causa do erro 400 do Groq)
- TTL de 1 hora por sessão
"""

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from src.config import settings


def _sanitizar_mensagens(mensagens: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    """
    Remove AIMessages com tool_calls sem ToolMessage de resposta correspondente.

    Por que isso acontece?
    Quando o Groq retorna erro 400 (tool_use_failed), o LangChain já salvou
    a AIMessage com tool_calls no Redis, mas o ToolMessage de resposta nunca
    chegou. Na próxima chamada, o Groq recebe um histórico inválido e falha
    de novo — causando o loop de erros que você estava vendo.

    Retorna: (lista_limpa, quantidade_removida)
    """
    limpas: list[BaseMessage] = []
    removidas = 0
    i = 0

    while i < len(mensagens):
        msg = mensagens[i]

        # Detecta AIMessage com tool_calls pendentes
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # Coleta todos os tool_call_ids desta mensagem
            ids_esperados = {tc["id"] for tc in msg.tool_calls if "id" in tc}

            if ids_esperados:
                # Verifica se existe um ToolMessage correspondente logo depois
                proximas = mensagens[i + 1 : i + 1 + len(ids_esperados)]
                ids_respondidos = {
                    m.tool_call_id
                    for m in proximas
                    if isinstance(m, ToolMessage) and hasattr(m, "tool_call_id")
                }

                if not ids_esperados.issubset(ids_respondidos):
                    # Tool call órfã — descarta AIMessage e os ToolMessages parciais
                    removidas += 1
                    i += 1
                    # Pula quaisquer ToolMessages parciais que vieram depois
                    while i < len(mensagens) and isinstance(mensagens[i], ToolMessage):
                        removidas += 1
                        i += 1
                    continue

        limpas.append(msg)
        i += 1

    return limpas, removidas


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    Retorna o histórico Redis sanitizado e com janela deslizante.

    Ordem das operações (importa!):
    1. Conecta ao Redis
    2. Sanitiza tool_calls órfãos  ← corrige o erro 400 do Groq
    3. Aplica sliding window de 20 mensagens
    4. Persiste o histórico limpo se houve mudanças
    """
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=3600,  # 1 hora de sessão
    )

    try:
        mensagens = history.messages

        if not mensagens:
            return history

        # --- Passo 1: Sanitiza tool_calls órfãos ---
        mensagens_limpas, n_removidas = _sanitizar_mensagens(mensagens)

        if n_removidas > 0:
            print(f"🧹 [{session_id}] {n_removidas} mensagem(ns) corrompida(s) removida(s) do histórico.")

        # --- Passo 2: Sliding window — mantém últimas 20 ---
        # ATENÇÃO: nunca corte no meio de um par AIMessage+ToolMessage.
        # A função abaixo garante que o corte acontece sempre em um
        # ponto seguro (início de um turno humano).
        if len(mensagens_limpas) > 20:
            candidatas = mensagens_limpas[-20:]

            # Ajusta para começar em uma HumanMessage (nunca no meio de um par tool)
            from langchain_core.messages import HumanMessage
            inicio_seguro = 0
            for j, m in enumerate(candidatas):
                if isinstance(m, HumanMessage):
                    inicio_seguro = j
                    break

            mensagens_limpas = candidatas[inicio_seguro:]

        # --- Passo 3: Persiste apenas se algo mudou ---
        if len(mensagens_limpas) != len(mensagens):
            history.clear()
            history.add_messages(mensagens_limpas)

    except Exception as e:
        print(f"⚠️  Erro ao processar histórico Redis [{session_id}]: {e}")

    return history


def limpar_historico(session_id: str) -> bool:
    """
    Apaga todo o histórico de uma sessão.
    Use no comando 'reiniciar' ou 'voltar' do bot.
    Retorna True se limpou com sucesso.
    """
    try:
        history = RedisChatMessageHistory(
            session_id=session_id,
            url=settings.REDIS_URL,
        )
        history.clear()
        print(f"🗑️  Histórico da sessão [{session_id}] apagado.")
        return True
    except Exception as e:
        print(f"⚠️  Falha ao limpar histórico [{session_id}]: {e}")
        return False