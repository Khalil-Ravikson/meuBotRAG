import uuid
import numpy as np
from typing import Optional

# Imports ignorados pelo Pylance para não sujar o ecrã
from redis.commands.search.query import Query  # type: ignore
# ── CORREÇÃO DO REDIS-PY ────────────────────────────────────────────────
try:
    # Para versões antigas do redis-py
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType  # type: ignore
except ImportError:
    # Para versões novas do redis-py (>= 5.3.x)
    from redis.commands.search.index_definition import IndexDefinition, IndexType  # type: ignore
# ────────────────────────────────────────────────────────────────────────
from redis.commands.search.field import TextField, VectorField, TagField  # type: ignore

# Imports corretos do teu projeto
from src.infrastructure.redis_client import get_redis
from src.rag.embeddings import get_embeddings

# ────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES DO CACHE
# ────────────────────────────────────────────────────────────────────────

CACHE_INDEX_NAME = "idx:semantic_cache"
CACHE_PREFIX = "cache:"
SIMILARITY_THRESHOLD = 0.95  # 95% de similaridade
CACHE_TTL = 86400 * 7        # 7 dias
VECTOR_DIMENSION = 1024      # Dimensão correta do BAAI/bge-m3

def init_cache_index():
    """Inicializa o índice do RedisSearch de forma síncrona."""
    redis_client = get_redis()
    try:
        redis_client.ft(CACHE_INDEX_NAME).info()
    except Exception:
        schema = (
            TagField("doc_type"),
            TextField("query_text"),
            TextField("response"),
            VectorField(
                "vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIMENSION,
                    "DISTANCE_METRIC": "COSINE"
                }
            )
        )
        definition = IndexDefinition(prefix=[CACHE_PREFIX], index_type=IndexType.HASH)
        redis_client.ft(CACHE_INDEX_NAME).create_index(fields=schema, definition=definition)
        print("✅ Índice de Semantic Cache criado no Redis!")

# Garante que o índice existe ao importar o módulo
init_cache_index()

# ────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE USO SÍNCRONAS
# ────────────────────────────────────────────────────────────────────────

def check_cache(query: str, doc_type: str, threshold: float = SIMILARITY_THRESHOLD) -> Optional[str]:
    try:
        redis_client = get_redis()
        embeddings_model = get_embeddings()
        
        # Obter embedding síncrono da query
        query_vector = embeddings_model.embed_query(query) 
        vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

        redis_query = (
            Query(f"(@doc_type:{{{doc_type}}})=>[KNN 1 @vector $vec AS score]")
            .sort_by("score")
            .return_fields("response", "score", "query_text")
            .dialect(2)
        )

        results = redis_client.ft(CACHE_INDEX_NAME).search(
            redis_query, 
            query_params={"vec": vector_bytes}
        )

        if results.docs:
            doc = results.docs[0]
            distance = float(doc.score)
            similarity = 1.0 - distance

            if similarity >= threshold:
                print(f"🎯 Cache HIT! Similaridade: {similarity:.4f} | Original: {doc.query_text}")
                return doc.response

    except Exception as e:
        print(f"❌ Erro ao buscar no Semantic Cache: {e}")
    
    return None

def store_cache(query: str, response: str, doc_type: str, ttl: int = CACHE_TTL):
    try:
        redis_client = get_redis()
        embeddings_model = get_embeddings()
        
        query_vector = embeddings_model.embed_query(query)
        vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        
        cache_id = f"{CACHE_PREFIX}{uuid.uuid4().hex}"
        
        payload = {
            "doc_type": doc_type,
            "query_text": query,
            "response": response,
            "vector": vector_bytes
        }
        
        redis_client.hset(cache_id, mapping=payload)
        redis_client.expire(cache_id, ttl)
        
        print(f"💾 Resposta salva no Semantic Cache! Rota: {doc_type}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar no Semantic Cache: {e}")