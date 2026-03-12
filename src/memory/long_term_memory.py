"""
memory/long_term_memory.py — Memória Factual de Longo Prazo (v3.1 — Fix Crítico)
==================================================================================

CORRECÇÃO APLICADA (v3.0 → v3.1):
────────────────────────────────────
  BUG CRÍTICO que causava:
    ⚠️ Falha embedding pergunta para fatos: No module named 'src.rag.vector_store'

  CAUSA: guardar_fato() e buscar_fatos_relevantes() continham importação lazy
    de 'src.rag.vector_store' — ficheiro eliminado na migração pgvector→Redis.
    O try/except engolia o ModuleNotFoundError, fazendo fallback silencioso
    para buscar_fatos_recentes() (sem semântica). Resultado: fatos do aluno
    nunca usados na personalização da resposta.

  CORRECÇÃO: Todas as referências substituídas por 'src.rag.embeddings'.

O QUE SÃO FATOS:
─────────────────
  Fragmentos de conhecimento sobre o aluno extraídos das conversas:
    - "Aluno do curso de Engenharia Civil, turno noturno"
    - "Inscrito no PAES 2026 na categoria BR-PPI"
    - "Dúvida recorrente sobre trancamento de matrícula"

ARQUITECTURA REDIS:
────────────────────
  mem:facts:list:{user_id}        → Lista LPUSH (Quick Recall)
  mem:facts:vec:{user_id}:{hash}  → JSON com texto + embedding (Semantic Recall)
  TTL: 30 dias em ambas as estruturas
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from src.infrastructure.redis_client import VECTOR_DIM, get_redis, get_redis_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

_TTL_FATOS             = 86400 * 30  # 30 dias
_MAX_FATOS_USER        = 50          # limite por utilizador
_MAX_FATOS_QUERY       = 5           # máximo retornado por busca
_PREFIX_FATOS_LIST     = "mem:facts:list:"
_PREFIX_FATOS_VEC      = "mem:facts:vec:"
_THRESHOLD_RELEVANCIA  = 0.65        # limiar de similaridade coseno


# ─────────────────────────────────────────────────────────────────────────────
# Tipo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Fato:
    texto:     str
    user_id:   str
    timestamp: int   = 0
    score:     float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = int(time.time())

    @property
    def hash_id(self) -> str:
        return hashlib.md5(self.texto.lower().strip().encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Guardar
# ─────────────────────────────────────────────────────────────────────────────

def guardar_fato(user_id: str, texto_fato: str) -> bool:
    """
    Embeda e guarda um fato no Redis (lista + JSON vectorial).
    Retorna True se novo, False se duplicado ou erro.
    """
    if not texto_fato or not texto_fato.strip():
        return False

    texto = texto_fato.strip()
    fato  = Fato(texto=texto, user_id=user_id)
    key_v = f"{_PREFIX_FATOS_VEC}{user_id}:{fato.hash_id}"
    r_txt = get_redis_text()

    # Evita duplicados
    try:
        if r_txt.exists(key_v):
            return False
    except Exception:
        pass

    # ── Embedding (CORRECÇÃO: src.rag.embeddings, não src.rag.vector_store) ──
    vetor: list[float] = []
    try:
        from src.rag.embeddings import get_embeddings  # ← CORRIGIDO
        vetor = get_embeddings().embed_query(texto)
    except Exception as e:
        logger.warning("⚠️  Sem embedding para fato [%s]: %s", user_id, e)

    # ── Guarda JSON vectorial ─────────────────────────────────────────────────
    try:
        r_txt.json().set(key_v, "$", {
            "texto":     texto,
            "user_id":   user_id,
            "timestamp": fato.timestamp,
            "embedding": vetor,
        })
        r_txt.expire(key_v, _TTL_FATOS)
    except Exception as e:
        logger.error("❌ Falha ao guardar fato [%s]: %s", user_id, e)
        return False

    # ── Guarda lista Quick Recall ─────────────────────────────────────────────
    key_l = f"{_PREFIX_FATOS_LIST}{user_id}"
    try:
        r_txt.lpush(key_l, texto)
        r_txt.ltrim(key_l, 0, _MAX_FATOS_USER - 1)
        r_txt.expire(key_l, _TTL_FATOS)
    except Exception as e:
        logger.warning("⚠️  Falha lista fatos [%s]: %s", user_id, e)

    logger.info("💾 Fato guardado [%s]: %.80s", user_id, texto)
    return True


def guardar_fatos_batch(user_id: str, fatos: list[str]) -> int:
    """Guarda múltiplos fatos. Retorna total de novos guardados."""
    n = sum(1 for t in fatos if guardar_fato(user_id, t))
    if n:
        logger.info("💾 Batch: %d/%d fatos novos [%s]", n, len(fatos), user_id)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Buscar
# ─────────────────────────────────────────────────────────────────────────────

def buscar_fatos_relevantes(
    user_id: str,
    pergunta: str,
    max_fatos: int = _MAX_FATOS_QUERY,
) -> list[Fato]:
    """
    Busca semântica local: embed da pergunta → coseno contra todos os fatos.

    CORRECÇÃO: from src.rag.embeddings import get_embeddings  (não vector_store)
    """
    try:
        from src.rag.embeddings import get_embeddings  # ← CORRIGIDO
        vetor_p = get_embeddings().embed_query(pergunta)
    except Exception as e:
        logger.warning("⚠️  Falha embedding pergunta para fatos: %s", e)
        return buscar_fatos_recentes(user_id, max_fatos)

    r = get_redis()
    fatos_raw = _carregar_fatos_com_embedding(r, user_id)
    if not fatos_raw:
        return buscar_fatos_recentes(user_id, max_fatos)

    pares: list[tuple[float, Fato]] = []
    for item in fatos_raw:
        vetor_f = item.get("embedding", [])
        if not vetor_f:
            continue
        score = _cosseno(vetor_p, vetor_f)
        if score >= _THRESHOLD_RELEVANCIA:
            pares.append((score, Fato(
                texto=item.get("texto", ""),
                user_id=user_id,
                timestamp=item.get("timestamp", 0),
                score=score,
            )))

    pares.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in pares[:max_fatos]]


def buscar_fatos_recentes(user_id: str, max_fatos: int = _MAX_FATOS_QUERY) -> list[Fato]:
    """Fallback rápido: N fatos mais recentes sem semântica."""
    r_txt = get_redis_text()
    try:
        textos = r_txt.lrange(f"{_PREFIX_FATOS_LIST}{user_id}", 0, max_fatos - 1)
        return [Fato(texto=t, user_id=user_id) for t in textos if t]
    except Exception:
        return []


def fatos_como_string(fatos: list[Fato]) -> str:
    if not fatos:
        return ""
    return "\n".join(f"- {f.texto}" for f in fatos)


def listar_todos_fatos(user_id: str) -> list[str]:
    r_txt = get_redis_text()
    try:
        return r_txt.lrange(f"{_PREFIX_FATOS_LIST}{user_id}", 0, _MAX_FATOS_USER - 1) or []
    except Exception:
        return []


def apagar_fatos(user_id: str) -> None:
    r, r_txt = get_redis(), get_redis_text()
    r_txt.delete(f"{_PREFIX_FATOS_LIST}{user_id}")
    cur, deleted = 0, 0
    while True:
        cur, keys = r.scan(cur, match=f"{_PREFIX_FATOS_VEC}{user_id}:*", count=100)
        if keys:
            r.delete(*keys)
            deleted += len(keys)
        if cur == 0:
            break
    logger.info("🗑️  Fatos apagados [%s]: lista + %d vetores", user_id, deleted)


# ─────────────────────────────────────────────────────────────────────────────
# Internos
# ─────────────────────────────────────────────────────────────────────────────

def _carregar_fatos_com_embedding(r, user_id: str) -> list[dict]:
    pattern, fatos, cur = f"{_PREFIX_FATOS_VEC}{user_id}:*", [], 0
    while True:
        cur, keys = r.scan(cur, match=pattern, count=100)
        for key in keys:
            try:
                doc = r.json().get(key, "$")
                if doc:
                    fatos.append(doc[0] if isinstance(doc, list) else doc)
            except Exception:
                pass
        if cur == 0:
            break
    return fatos


def _cosseno(v1: list[float], v2: list[float]) -> float:
    """Produto escalar para vetores normalizados (BAAI/bge-m3 com normalize=True)."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(v1, v2))))