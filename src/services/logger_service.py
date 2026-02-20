"""
================================================================================
logger_service.py — Serviço de Log Centralizado (v2)
================================================================================

MELHORIAS v2:
  1. log_error() agora também usa logging.getLogger (aparece no terminal limpo)
  2. log_info() adicionado para eventos importantes (não só erros)
  3. log_warn() para alertas não críticos
  4. get_recent_errors() para debug via endpoint /logs
  5. Bare except removido → exception específica com fallback
  6. Prefixos de emoji padronizados por nível
================================================================================
"""

import json
import logging
import redis
from datetime import datetime
from src.config import settings

logger = logging.getLogger(__name__)


class LogService:
    def __init__(self):
        try:
            self.r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.r.ping()
        except Exception as e:
            self.r = None
            logger.warning("⚠️  LogService: Redis indisponível. Logs só no terminal. Erro: %s", e)

    def _salvar_redis(self, nivel: str, user_id: str, context: str, msg: str):
        """Salva entrada de log no Redis com TTL implícito via ltrim."""
        if not self.r:
            return
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "nivel":     nivel,
                "user":      user_id,
                "context":   context,
                "msg":       str(msg)[:500],  # limita tamanho para não lotar
            }
            chave = f"system_logs:{nivel.lower()}"
            self.r.lpush(chave, json.dumps(payload))
            self.r.ltrim(chave, 0, 99)  # mantém últimos 100 por nível
        except Exception as e:
            logger.debug("LogService: falha ao salvar no Redis: %s", e)

    def log_error(self, user_id: str, context: str, error_msg: str):
        """Registra erro crítico. Aparece no terminal e no Redis."""
        logger.error("❌ [%s] %s | %s", user_id, context, str(error_msg)[:200])
        self._salvar_redis("ERROR", user_id, context, error_msg)

    def log_warn(self, user_id: str, context: str, msg: str):
        """Registra aviso não crítico."""
        logger.warning("⚠️  [%s] %s | %s", user_id, context, str(msg)[:200])
        self._salvar_redis("WARN", user_id, context, msg)

    def log_info(self, user_id: str, context: str, msg: str):
        """Registra evento informativo importante (não aparece em produção por padrão)."""
        logger.info("ℹ️  [%s] %s | %s", user_id, context, str(msg)[:200])
        self._salvar_redis("INFO", user_id, context, msg)

    def get_recent_errors(self, limit: int = 20) -> list:
        """Retorna os últimos N erros do Redis (para endpoint /logs)."""
        if not self.r:
            return []
        try:
            raw = self.r.lrange("system_logs:error", 0, limit - 1)
            return [json.loads(e) for e in raw]
        except Exception:
            return []