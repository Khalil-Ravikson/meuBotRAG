"""
rag/embeddings.py — Modelo de Embedding Local (substitui rag/vector_store.py)
==============================================================================

POR QUÊ ESTE FICHEIRO EXISTE:
───────────────────────────────
  O ficheiro original src/rag/vector_store.py tinha duas responsabilidades:
    1. get_vector_store() → ligação ao pgvector (PostgreSQL)   ← ELIMINADO
    2. get_embeddings()   → modelo BAAI/bge-m3 local (CPU)    ← MANTIDO AQUI

  Com a migração para Redis Stack, o pgvector foi eliminado.
  O modelo de embedding continua local e necessário — é ele que converte
  texto em vectores de 1024 dimensões para o Redis HNSW e o BM25 não precisam.

  Este ficheiro é o ÚNICO lugar onde o modelo é carregado.
  Todos os outros módulos importam daqui:

    from src.rag.embeddings import get_embeddings

  Ficheiros que importam get_embeddings():
    - src/rag/ingestion.py          (gera embeddings dos chunks na ingestão)
    - src/rag/hybrid_retriever.py   (embedding da query para busca vectorial)
    - src/domain/semantic_router.py (embedding da mensagem para routing)
    - src/memory/long_term_memory.py (embedding dos fatos para KNN)
    - src/tools/calendar_tool.py    (embedding da query antes da busca)
    - src/tools/tool_edital.py      (idem)
    - src/tools/tool_contatos.py    (idem)

SOBRE O MODELO BAAI/bge-m3:
─────────────────────────────
  - Tamanho: ~1.3GB (descarregado automaticamente no primeiro arranque)
  - Dimensões: 1024 floats por vector
  - Língua: multilíngue (PT, EN, ES, e mais 100+)
  - Device: CPU (a RX 580 não tem suporte CUDA nativo fácil no Windows/Docker)
  - normalize_embeddings=True: obrigatório para similaridade coseno correcta

  NOTA SOBRE RAM:
    O modelo ocupa ~1.3GB de RAM quando carregado.
    O @lru_cache garante que é carregado UMA SÓ VEZ por processo.
    Todos os módulos que fazem `get_embeddings()` recebem a mesma instância.
    Custo total: 1.3GB fixo, independente de quantas tools ou módulos existem.

SOBRE HF_TOKEN:
────────────────
  Sem token: download anónimo (sujeito a rate limit do HuggingFace Hub)
  Com token: download autenticado (mais rápido, sem rate limit)
  O token NÃO afecta a velocidade de inferência (isso é CPU/GPU).
  Obtém em: https://huggingface.co/settings/tokens
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

_MODELO = "BAAI/bge-m3"
_DIMS   = 1024   # Dimensões do vector — deve bater com VECTOR_DIM no redis_client.py


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Retorna o modelo de embedding singleton (carregado uma única vez).

    O @lru_cache garante que o modelo de 1.3GB é carregado uma só vez
    por processo, independentemente de quantos módulos chamem esta função.

    Thread-safe: o lru_cache do Python é thread-safe para leituras.
    """
    # Configura HF_TOKEN para download autenticado (evita rate limit)
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"]                  = settings.HF_TOKEN
        os.environ["HUGGING_FACE_HUB_TOKEN"]    = settings.HF_TOKEN
        logger.info("🔑 HF_TOKEN configurado — download autenticado.")
    else:
        logger.info("⚠️  HF_TOKEN ausente — download anónimo (pode ser lento na 1ª vez).")

    logger.info("🔄 A carregar modelo de embedding: %s (CPU, ~1.3GB)...", _MODELO)

    model = HuggingFaceEmbeddings(
        model_name=_MODELO,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("✅ Modelo '%s' pronto. Dimensões: %d.", _MODELO, _DIMS)
    return model


def get_dims() -> int:
    """Retorna o número de dimensões do modelo actual. Útil para criar índices Redis."""
    return _DIMS