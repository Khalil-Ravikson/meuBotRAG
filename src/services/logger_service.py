import redis
import json
from datetime import datetime
from src.config import settings

class LogService:
    def __init__(self):
        try:
            self.r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        except:
            self.r = None

    def log_error(self, user_id: str, context: str, error_msg: str):
        """Salva o erro no Redis sem parar o bot"""
        if not self.r:
            print(f"⚠️ Redis Off. Erro não salvo: {error_msg}")
            return

        payload = {
            "timestamp": datetime.now().isoformat(),
            "user": user_id,
            "context": context,
            "error": str(error_msg)
        }
        
        # Empilha o erro na lista 'system_errors'
        self.r.lpush("system_logs:errors", json.dumps(payload))
        # Mantém apenas os últimos 100 erros para não lotar memória
        self.r.ltrim("system_logs:errors", 0, 99)
        
        print(f"🔥 Erro Registrado no Redis: {context} - {error_msg}")