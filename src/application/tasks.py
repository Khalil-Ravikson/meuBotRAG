import logging
import asyncio
from src.infrastructure.celery_app import celery_app
from src.application.handle_message import handle_message 
from src.infrastructure.redis_client import get_redis  # Importe a sua ligação ao Redis

logger = logging.getLogger(__name__)

def run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

@celery_app.task(name="processar_mensagem_whatsapp", bind=True, max_retries=3)
def processar_mensagem_task(self, identity: dict):
    chat_id = identity.get("chat_id")
    
    # 1. Pega a instância do Redis (usando a mesma que você já tem no projeto)
    redis_client = get_redis() 
    
    # 2. Configura o Cadeado (Lock) para este usuário específico
    # - timeout=60: Se o bot crachar feio, a porta destranca sozinha em 60s
    # - blocking_timeout=2: Espera no máximo 2s para ver se a porta abre
    lock_name = f"lock:chat:{chat_id}"
    lock = redis_client.lock(lock_name, timeout=60, blocking_timeout=2)
    
    # 3. Tenta pegar a chave da porta
    acquired = lock.acquire()
    
    if not acquired:
        # Se a porta está trancada, significa que o bot já está respondendo a este usuário!
        logger.warning("🔒 Usuário %s já está em atendimento. Reenfileirando mensagem para daqui a 3s...", chat_id)
        # O Celery joga a mensagem de volta pra fila e tenta de novo em 3 segundos
        raise self.retry(countdown=3)

    # 4. Se chegou aqui, a porta está destrancada e somos os donos do cadeado!
    try:
        logger.info("👷 [CELERY] Iniciando processamento exclusivo para %s", chat_id)
        run_async(handle_message(identity))
        logger.info("✅ [CELERY] Processamento concluído para %s", chat_id)
        
    except Exception as exc:
        logger.error("❌ [CELERY] Erro ao processar %s: %s", chat_id, exc)
        raise self.retry(exc=exc, countdown=5)
        
    finally:
        # 5. REGRA DE OURO: Sempre destrancar a porta no final, aconteça o que acontecer
        try:
            lock.release()
            logger.debug("🔓 Lock liberado para o usuário %s", chat_id)
        except Exception:
            pass # Ignora se o lock já tiver expirado pelo timeout de 60s