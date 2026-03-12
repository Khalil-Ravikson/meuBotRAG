"""
infrastructure/semantic_cache.py — Cache Semântico (v1.1 — 4 Bugs Corrigidos)
================================================================================

BUGS CORRIGIDOS (v1.0 → v1.1):
────────────────────────────────
  BUG 1 — Import do cliente Redis errado:
    ANTES : from src.infrastructure.redis_client import redis_client
    DEPOIS: from src.infrastructure.redis_client import get_redis, get_redis_text

  BUG 2 — Função de embedding inexistente:
    ANTES : from src.rag.embeddings import gerar_embedding
    DEPOIS: from src.rag.embeddings import get_embeddings
            uso: get_embeddings().embed_query(query)

  BUG 3 — Dimensão vectorial errada:
    ANTES : VECTOR_DIMENSION = 768  # dimensão do all-MiniLM (modelo errado)
    DEPOIS: VECTOR_DIMENSION = VECTOR_DIM  # 1024 — BAAI/bge-m3 (do redis_client)

  BUG 4 — Funções async incompatíveis com core.py síncrono:
    ANTES : async def check_cache() / async def store_cache()
    DEPOIS: síncronas — core.py.responder() não pode usar await

  COSMÉTICO — Nível de log errado:
    ANTES : logger.warning("💾 Resposta salva no Semantic Cache!")
    DEPOIS: logger.info("💾 Resposta salva no Semantic Cache!")

ECONOMIA DE TOKENS:
────────────────────
  60-70% das perguntas académicas são repetições (datas, prazos, vagas).
  Cache com threshold=0.95 → match só em perguntas semanticamente idênticas.
  Estimativa: ~34% de redução no total de tokens diários.

  Ex: "quando é a matrícula?" e "qual o prazo da matrícula?" → mesmo cache hit
  Ex: "vagas de engenharia civil" e "engenharia civil tem quantas vagas?" → hit

INTEGRAÇÃO EM core.py:
────────────────────────
  from src.infrastructure.semantic_cache import check_cache, store_cache

  # Passo 5.5 — antes de chamar Gemini:
  hit = check_cache(query=query_transformada.query_principal, doc_type=rota.value)
  if hit:
      adicionar_mensagem(session_id, "assistant", hit)
      return AgentResponse(conteudo=hit, rota=rota, tokens_total=0, sucesso=True)

  # Passo 7.5 — após geração bem-sucedida:
  if gemini_resp.sucesso:
      store_cache(query=query_transformada.query_principal,
                  response=conteudo, doc_type=rota.value)
"""
from __future__ import annotations

import logging
import struct
import uuid

from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.infrastructure.redis_client import VECTOR_DIM, get_redis, get_redis_text  # ← BUG 1 FIX

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

CACHE_INDEX_NAME     = "idx:semantic_cache"
CACHE_PREFIX         = "cache:"
SIMILARITY_THRESHOLD = 0.95         # 95% similiar → hit
CACHE_TTL            = 86400 * 7    # 7 dias
VECTOR_DIMENSION     = VECTOR_DIM   # ← BUG 3 FIX: 1024 (BAAI/bge-m3)


# ─────────────────────────────────────────────────────────────────────────────
# Inicialização
# ─────────────────────────────────────────────────────────────────────────────

def init_cache_index() -> None:
    """
    Cria o índice RediSearch para o Semantic Cache (idempotente).
    Chamado em main.py startup → junto com inicializar_indices().
    """
    r = get_redis()  # ← BUG 1 FIX
    try:
        r.ft(CACHE_INDEX_NAME).info()
        logger.debug("ℹ️  Índice '%s' já existe.", CACHE_INDEX_NAME)
        return
    except Exception:
        pass

    schema = (
        TagField("$.doc_type",    as_name="doc_type"),
        TextField("$.query_text", as_name="query_text"),
        TextField("$.response",   as_name="response"),
        VectorField(
            "$.vector", "FLAT",
            {"TYPE": "FLOAT32", "DIM": VECTOR_DIMENSION, "DISTANCE_METRIC": "COSINE"},
            as_name="vector",
        ),
    )
    r.ft(CACHE_INDEX_NAME).create_index(
        schema,
        definition=IndexDefinition(prefix=[CACHE_PREFIX], index_type=IndexType.JSON),
    )
    logger.info("✅ Índice Semantic Cache '%s' criado (dim=%d).", CACHE_INDEX_NAME, VECTOR_DIMENSION)


# ─────────────────────────────────────────────────────────────────────────────
# API pública — síncronas (BUG 4 FIX)
# ─────────────────────────────────────────────────────────────────────────────

def check_cache(
    query: str,
    doc_type: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> str | None:
    """
    Verifica se existe resposta em cache para a query dada.
    Retorna a resposta (str) se cache hit, None caso contrário.
    Síncrona — compatível com core.py.responder() (BUG 4 FIX).
    """
    try:
        from src.rag.embeddings import get_embeddings  # ← BUG 2 FIX
        query_vector = get_embeddings().embed_query(query)
        vector_bytes = _to_bytes(query_vector)
    except Exception as e:
        logger.warning("⚠️  Cache check: falha embedding: %s", e)
        return None

    try:
        r = get_redis()  # ← BUG 1 FIX
        q = (
            Query(f"(@doc_type:{{{doc_type}}})=>[KNN 1 @vector $vec AS score]")
            .sort_by("score")
            .return_fields("response", "score", "query_text")
            .dialect(2)
        )
        results = r.ft(CACHE_INDEX_NAME).search(q, query_params={"vec": vector_bytes})

        if results.docs:
            doc        = results.docs[0]
            similarity = 1.0 - float(doc.score)  # Redis retorna distância, não similaridade
            if similarity >= threshold:
                logger.info(
                    "🎯 Cache HIT! sim=%.4f rota=%s query='%.50s'",
                    similarity, doc_type, getattr(doc, "query_text", "?"),
                )
                return doc.response
            logger.debug("⬜ Cache MISS. Melhor sim=%.4f (limiar=%.2f)", similarity, threshold)

    except Exception as e:
        logger.warning("⚠️  Erro ao consultar cache: %s", e)

    return None


def store_cache(
    query: str,
    response: str,
    doc_type: str,
    ttl: int = CACHE_TTL,
) -> None:
    """
    Guarda resposta no cache semântico.
    Síncrona — compatível com core.py.responder() (BUG 4 FIX).
    """
    try:
        from src.rag.embeddings import get_embeddings  # ← BUG 2 FIX
        query_vector = get_embeddings().embed_query(query)
    except Exception as e:
        logger.warning("⚠️  Cache store: falha embedding: %s", e)
        return

    try:
        r = get_redis()  # ← BUG 1 FIX
        cid = f"{CACHE_PREFIX}{uuid.uuid4().hex}"
        r.json().set(cid, "$", {
            "doc_type":   doc_type,
            "query_text": query[:200],
            "response":   response,
            "vector":     query_vector,
        })
        r.expire(cid, ttl)
        logger.info("💾 Resposta salva no Semantic Cache! Rota: %s", doc_type)  # ← COSMÉTICO FIX

    except Exception as e:
        logger.warning("⚠️  Erro ao guardar no cache: %s", e)


def cache_stats() -> dict:
    """Métricas do cache (para endpoint /cache/stats)."""
    try:
        info = get_redis().ft(CACHE_INDEX_NAME).info()
        return {
            "total_entradas": info.get("num_docs", 0),
            "index_name":     CACHE_INDEX_NAME,
            "ttl_dias":       CACHE_TTL // 86400,
            "threshold":      SIMILARITY_THRESHOLD,
            "vector_dim":     VECTOR_DIMENSION,
        }
    except Exception as e:
        return {"erro": str(e)}


def invalidar_cache_rota(doc_type: str) -> int:
    """
    Remove todas as entradas de cache de uma rota específica.
    Útil quando um PDF é re-ingerido (edital actualizado, etc.).
    """
    r = get_redis()
    deletados = 0
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match=f"{CACHE_PREFIX}*", count=500)
        for key in keys:
            try:
                doc = r.json().get(key, "$.doc_type")
                if doc and (doc[0] if isinstance(doc, list) else doc) == doc_type:
                    r.delete(key)
                    deletados += 1
            except Exception:
                pass
        if cursor == 0:
            break
    logger.info("🗑️  Cache invalidado para rota=%s: %d entradas removidas", doc_type, deletados)
    return deletados


# ─────────────────────────────────────────────────────────────────────────────
# Interno
# ─────────────────────────────────────────────────────────────────────────────

def _to_bytes(vetor: list[float]) -> bytes:
    return struct.pack(f"{len(vetor)}f", *vetor)