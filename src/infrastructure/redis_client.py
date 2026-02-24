"""
infrastructure/redis_client.py — Cliente Redis singleton
=========================================================
Um único pool de conexões compartilhado por memory/, cache/ e middleware/.
"""
from __future__ import annotations
import logging
import redis
from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)
_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        try:
            _client.ping()
            logger.info("✅ Redis conectado: %s", settings.REDIS_URL)
        except redis.ConnectionError as e:
            logger.error("❌ Redis offline: %s", e)
            raise
    return _client


def redis_ok() -> bool:
    try:
        get_redis().ping()
        return True
    except Exception:
        return False