from langchain_community.chat_message_histories import RedisChatMessageHistory
from src.config import settings

# O nome TEM que ser este, pois é o que o rag_service importa
def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    Retorna o histórico do Redis com auto-limpeza (trimming).
    Mantém apenas as últimas 30 mensagens para evitar estouro de tokens
    e manter o bot rápido, sem precisar de uma LLM secundária.
    """
    
    # 1. Conexão com o Redis
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=3600 # 1 hora de sessão
    )

    # 2. Estratégia de "Janela Deslizante" (Sliding Window)
    # Se tiver muitas mensagens, cortamos as antigas manualmente.
    # Isso é mais rápido e seguro que usar SummaryMemory com Agentes novos.
    try:
        mensagens_atuais = history.messages
        if len(mensagens_atuais) > 20:
            # Pega apenas as últimas 20 mensagens
            ultimas_20 = mensagens_atuais[-20:]
            
            # Limpa o banco para essa sessão
            history.clear()
            
            # Reinsere apenas as recentes
            for msg in ultimas_20:
                history.add_message(msg)
                
    except Exception as e:
        print(f"⚠️ Erro leve ao limpar histórico Redis: {e}")

    # Retorna o objeto HISTORY direto (que é o que o Runnable espera)
    return history