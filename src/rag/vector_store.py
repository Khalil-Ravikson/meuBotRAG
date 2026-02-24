"""
rag/vector_store.py — Embedding singleton + interface pgvector
==============================================================
Substitui src/services/db_service.py.

SOBRE O HF_TOKEN:
  Configura o token do HuggingFace Hub antes de carregar o modelo.
  Benefícios:
    - Evita rate limit no download (sem token = limite anônimo por IP)
    - Acesso a modelos privados/gated no futuro
    - Download mais rápido via CDN autenticado
  Limitação: NÃO acelera a inferência — isso depende de CPU/GPU.

SOBRE O MODELO BAAI/bge-m3:
  ~1.3GB, multilíngue, bom para português.
  Carregado UMA vez via @lru_cache — sem custo ao chamar get_vector_store()
  de múltiplas tools.

ATENÇÃO ao collection_name:
  Se você tinha dados com "receitas_bot", mantenha esse nome até re-ingerir.
"""
from __future__ import annotations
import logging
import os
from functools import lru_cache

from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "BAAI/bge-m3"
_COLLECTION_NAME = "uema_bot"   # ⚠️ trocar se banco antigo usa "receitas_bot"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Singleton do modelo de embedding (~1.3GB carregado uma vez).
    HF_TOKEN configurado via env var, que é como o Hub espera receber.
    """
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"] = settings.HF_TOKEN
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN
        logger.info("🔑 HF_TOKEN configurado — download autenticado.")
    else:
        logger.info("⚠️  HF_TOKEN ausente — download anônimo (pode ser lento).")

    logger.info("🔄 Carregando modelo de embedding: %s", _EMBEDDING_MODEL)
    emb = HuggingFaceEmbeddings(
        model_name=_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},               # troque para "cuda" se tiver GPU
        encode_kwargs={"normalize_embeddings": True}, # melhora similaridade coseno
    )
    logger.info("✅ Embedding pronto: %s", _EMBEDDING_MODEL)
    return emb


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    """Singleton do banco vetorial. Conecta ao pgvector uma única vez."""
    try:
        vs = PGVector(
            embeddings=get_embeddings(),
            collection_name=_COLLECTION_NAME,
            connection=settings.DATABASE_URL,
            use_jsonb=True,
        )
        logger.info("✅ pgvector conectado | coleção: %s", _COLLECTION_NAME)
        return vs
    except Exception as e:
        logger.error("❌ Falha ao conectar pgvector: %s", e)
        raise RuntimeError(f"pgvector indisponível: {e}") from e


def diagnosticar() -> set[str]:
    """Retorna sources únicos no banco. Use quando tools retornam 'Não encontrei'."""
    try:
        docs = get_vector_store().similarity_search("UEMA", k=50)
        sources = {doc.metadata.get("source", "SEM_SOURCE") for doc in docs}
        logger.info("🔍 Sources no banco: %s", sources)
        return sources
    except Exception as e:
        logger.error("❌ Diagnóstico falhou: %s", e)
        return set()