"""
infrastructure/observability.py — Métricas e logs estruturados
==============================================================
Substitui o logger_service.py anterior.
Salva logs no Redis e expõe métricas de tokens/latência/erros.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from src.infrastructure.redis_client import get_redis

logger = logging.getLogger(__name__)

_PREFIXOS = {
    "error": "system_logs:error",
    "warn":  "system_logs:warn",
    "info":  "system_logs:info",
}
_MAX_ENTRIES = 100


class Observability:
    """Singleton de observabilidade — logs + métricas de tokens."""

    def _salvar(self, nivel: str, user_id: str, context: str, msg: str) -> None:
        try:
            r = get_redis()
            entrada = json.dumps({
                "ts":      datetime.now().isoformat(),
                "user_id": user_id,
                "context": context,
                "msg":     str(msg)[:300],
            }, ensure_ascii=False)
            chave = _PREFIXOS.get(nivel, "system_logs:info")
            r.lpush(chave, entrada)
            r.ltrim(chave, 0, _MAX_ENTRIES - 1)
        except Exception:
            pass  # nunca quebra o fluxo principal

    def error(self, user_id: str, context: str, msg: str) -> None:
        logger.error("❌ [%s] %s | %s", user_id, context, str(msg)[:200])
        self._salvar("error", user_id, context, msg)

    def warn(self, user_id: str, context: str, msg: str) -> None:
        logger.warning("⚠️  [%s] %s | %s", user_id, context, str(msg)[:200])
        self._salvar("warn", user_id, context, msg)

    def info(self, user_id: str, context: str, msg: str) -> None:
        logger.info("ℹ️  [%s] %s | %s", user_id, context, str(msg)[:200])
        self._salvar("info", user_id, context, msg)

    def registrar_resposta(
        self,
        user_id: str,
        rota: str,
        tokens_entrada: int,
        tokens_saida: int,
        latencia_ms: int,
        iteracoes: int,
    ) -> None:
        """Salva métricas de uma resposta no Redis para análise posterior."""
        try:
            r = get_redis()
            entrada = json.dumps({
                "ts":             datetime.now().isoformat(),
                "user_id":        user_id,
                "rota":           rota,
                "tokens_entrada": tokens_entrada,
                "tokens_saida":   tokens_saida,
                "tokens_total":   tokens_entrada + tokens_saida,
                "latencia_ms":    latencia_ms,
                "iteracoes":      iteracoes,
            }, ensure_ascii=False)
            r.lpush("metrics:respostas", entrada)
            r.ltrim("metrics:respostas", 0, 499)
        except Exception:
            pass

    def get_recent_errors(self, limit: int = 20) -> list[dict]:
        try:
            r = get_redis()
            raw = r.lrange("system_logs:error", 0, limit - 1)
            return [json.loads(e) for e in raw]
        except Exception:
            return []

    def get_recent_metrics(self, limit: int = 50) -> list[dict]:
        try:
            r = get_redis()
            raw = r.lrange("metrics:respostas", 0, limit - 1)
            return [json.loads(e) for e in raw]
        except Exception:
            return []


# Singleton para importação direta
obs = Observability()