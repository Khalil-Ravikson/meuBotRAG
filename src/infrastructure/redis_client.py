"""
infrastructure/redis_client.py — Cliente Redis com Busca Híbrida (v3 — Redis Stack)
=====================================================================================

PORQUE SUBSTITUÍMOS O POSTGRESQL/pgvector POR REDIS:
─────────────────────────────────────────────────────
  Problema anterior:
    - pgvector → ~200MB de RAM mínimo só para o processo Postgres
    - Duas conexões de rede (app→postgres, app→redis) a gerir
    - Busca vetorial simples sem suporte a keyword filtering nativo
    - Container extra no docker-compose → mais RAM desperdiçada

  Solução com Redis Stack:
    - Redis em modo "stack" (redis/redis-stack:latest) inclui:
        RediSearch  → índice de texto completo (BM25) + vetor (HNSW/FLAT)
        RedisJSON   → armazena documentos como JSON nativo
    - UM SÓ processo, um só container
    - Busca Híbrida real: vetor E keyword na mesma query
    - Pipeline assíncrono nativo
    - Uso típico: ~60-80MB vs ~200MB+ do Postgres

COMO O REDIS STACK RESOLVE ALUCINAÇÕES:
────────────────────────────────────────
  Busca vetorial pura: "quando é matrícula?" → pode trazer chunk irrelevante
  Busca híbrida (vetor + BM25):
    - O vetor captura SEMÂNTICA ("quando ocorre o período de matrícula")
    - O BM25 captura KEYWORDS EXATAS ("matrícula", "2026.1", siglas "BR-PPI")
    - Os resultados são fundidos via Reciprocal Rank Fusion (RRF)
    - Resultado: chunks com datas e siglas exatas sobem para o topo

ESTRUTURA DOS ÍNDICES REDIS:
────────────────────────────
  rag:chunks:{source}:{id}   → RedisJSON com conteúdo + embedding
  tools:registry             → Hash com descrições das tools para roteamento semântico
  tools:embeddings:{name}    → Vector do texto da tool (para semantic routing)
  mem:working:{user_id}      → Hash com contexto ativo da conversa
  mem:facts:{user_id}        → List de fatos extraídos (Long-Term Memory)
  chat:{session_id}          → List do histórico de mensagens (sliding window)
  menu_state:{user_id}       → String com estado atual do menu
  user_ctx:{user_id}         → JSON com contexto persistente

ÍNDICES REDIS CRIADOS:
─────────────────────
  idx:rag:chunks  → VECTOR + TEXT search nos chunks dos PDFs
  idx:tools       → VECTOR search nas descrições das tools (roteamento semântico)
"""
from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from typing import Any

import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes dos índices
# ─────────────────────────────────────────────────────────────────────────────

# Dimensão do modelo BAAI/bge-m3 (mantemos o mesmo para compatibilidade)
# Se mudar o modelo, mude AQUI e re-ingira tudo.
VECTOR_DIM = 1024

# Prefixos das chaves no Redis
PREFIX_CHUNKS = "rag:chunk:"       # Chave: rag:chunk:{source}:{hash_id}
PREFIX_TOOLS  = "tools:emb:"       # Chave: tools:emb:{tool_name}
PREFIX_WORKING_MEM = "mem:work:"   # Working memory por sessão
PREFIX_FACTS  = "mem:facts:"       # Long-term factual memory
PREFIX_CHAT   = "chat:"            # Histórico de mensagens

# Nomes dos índices RediSearch
IDX_CHUNKS = "idx:rag:chunks"   # Índice híbrido para os chunks dos PDFs
IDX_TOOLS  = "idx:tools"        # Índice vetorial para semantic tool routing

# Parâmetros HNSW (equilibrio velocidade/precisão para hardware limitado)
# M=16       → número de links por nó (mais = melhor recall, mais RAM)
# EF=200     → tamanho da lista de candidatos na busca (mais = melhor recall, mais lento)
# EF_CONSTRUCTION=200 → qualidade do grafo na ingestão
HNSW_M  = 16
HNSW_EF = 200


# ─────────────────────────────────────────────────────────────────────────────
# Cliente Redis singleton
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    """
    Retorna cliente Redis singleton.

    ECONOMIA DE RAM:
      - lru_cache garante UMA só instância em todo o processo
      - connection pool reutiliza sockets (evita overhead de TCP handshake)
      - decode_responses=False → necessário para armazenar bytes dos embeddings
    """
    client = redis.Redis.from_url(
        settings.REDIS_URL,
        decode_responses=False,          # IMPORTANTE: False para suportar bytes (vetores)
        socket_connect_timeout=5,
        socket_timeout=10,
        retry_on_timeout=True,
        health_check_interval=30,
        max_connections=20,              # Pool limitado para hardware com pouca RAM
    )
    try:
        client.ping()
        logger.info("✅ Redis Stack conectado: %s", settings.REDIS_URL)
    except redis.ConnectionError as e:
        logger.error("❌ Redis offline: %s", e)
        raise RuntimeError(f"Redis indisponível: {e}") from e
    return client


def get_redis_text() -> redis.Redis:
    """
    Cliente Redis com decode_responses=True para operações de texto puro.
    Usado para: histórico, estado do menu, contexto do usuário.

    POR QUE DOIS CLIENTES?
      - get_redis()      → bytes, necessário para np.array serializado como vetor
      - get_redis_text() → str, mais ergonómico para manipular JSON/strings
    """
    # Reutiliza a URL mas cria cliente de texto
    # Não usamos lru_cache aqui para manter separação clara
    client = redis.Redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=10,
        retry_on_timeout=True,
    )
    return client


def redis_ok() -> bool:
    """Verifica saúde do Redis. Usado no /health endpoint."""
    try:
        get_redis().ping()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Criação e gestão dos índices RediSearch
# ─────────────────────────────────────────────────────────────────────────────

def criar_indice_chunks() -> None:
    """
    Cria o índice híbrido para os chunks dos PDFs.

    SCHEMA DO ÍNDICE idx:rag:chunks:
    ──────────────────────────────────
      $.content   → TextField (BM25 full-text, pt-br)
                    Permite busca por "matrícula 2026.1" sem embedding
      $.source    → TagField (filtragem exata por arquivo PDF)
                    Permite "@source:{edital_paes_2026}" na query
      $.chunk_id  → NumericField (para paginação e debugging)
      $.embedding → VectorField (HNSW, cosine similarity)
                    Permite busca semântica via KNN

    POR QUE INDEXTYPE.JSON?
      RedisJSON armazena documentos como JSON nativo com acesso por path $.
      Alternativa (HASH) exigiria serialização manual de cada campo.
      JSON é ~20% mais lento na ingestão, mas muito mais flexível para
      adicionar metadados hierárquicos futuramente.
    """
    r = get_redis()
    try:
        r.ft(IDX_CHUNKS).info()
        logger.info("ℹ️  Índice '%s' já existe.", IDX_CHUNKS)
        return
    except Exception:
        pass  # Índice não existe, vamos criar

    schema = (
        # ── Busca por texto completo (BM25) ──────────────────────────────────
        # NOSTEM → desativa stemming do inglês (pt-br não tem stemmer nativo)
        # WEIGHT=2.0 → conteúdo tem peso duplo vs outros campos
        TextField("$.content",  as_name="content",  no_stem=True, weight=2.0),
        TextField("$.source",   as_name="source_text"),   # Para busca por nome de arquivo

        # ── Filtragem exata ───────────────────────────────────────────────────
        TagField("$.source",    as_name="source"),         # @source:{edital_paes_2026.pdf}
        TagField("$.doc_type",  as_name="doc_type"),       # @doc_type:{calendario|edital|contatos}
        NumericField("$.chunk_index", as_name="chunk_idx"),

        # ── Busca vetorial (HNSW — Hierarchical Navigable Small World) ───────
        # HNSW é o melhor equilíbrio velocidade/precisão para datasets médios
        # FLAT seria exato mas O(n) — muito lento para >10k chunks
        VectorField(
            "$.embedding",
            "HNSW",
            {
                "TYPE":               "FLOAT32",
                "DIM":                VECTOR_DIM,
                "DISTANCE_METRIC":    "COSINE",    # Melhor para embeddings normalizados
                "M":                  HNSW_M,
                "EF_CONSTRUCTION":    HNSW_EF,
                "INITIAL_CAP":        5000,        # Reserva memória inicial
            },
            as_name="embedding",
        ),
    )

    r.ft(IDX_CHUNKS).create_index(
        schema,
        definition=IndexDefinition(
            prefix=[PREFIX_CHUNKS],
            index_type=IndexType.JSON,
        ),
    )
    logger.info("✅ Índice '%s' criado (Híbrido: BM25 + HNSW).", IDX_CHUNKS)


def criar_indice_tools() -> None:
    """
    Cria o índice vetorial para Semantic Tool Routing.

    COMO FUNCIONA O ROTEAMENTO SEMÂNTICO:
    ──────────────────────────────────────
      1. Na ingestão: cada tool tem uma descrição armazenada como JSON
         com o embedding da sua descrição pré-computado.
      2. Na query: convertemos a mensagem do usuário em vetor e buscamos
         qual tool tem a descrição mais similar — SEM chamar o LLM.
      3. Custo: ~0.5ms de CPU vs ~500ms de chamada ao Gemini.
         Isso representa economia de ~99.9% dos tokens de roteamento.

    SCHEMA:
      $.name        → Nome da tool (ex: "consultar_calendario_academico")
      $.description → Texto completo da descrição
      $.embedding   → Vetor da descrição (para KNN)
    """
    r = get_redis()
    try:
        r.ft(IDX_TOOLS).info()
        logger.info("ℹ️  Índice '%s' já existe.", IDX_TOOLS)
        return
    except Exception:
        pass

    schema = (
        TextField("$.name",        as_name="name"),
        TextField("$.description", as_name="description"),
        VectorField(
            "$.embedding",
            "FLAT",                    # FLAT é exato e eficiente para poucos registos (~10 tools)
            {
                "TYPE":            "FLOAT32",
                "DIM":             VECTOR_DIM,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="embedding",
        ),
    )

    r.ft(IDX_TOOLS).create_index(
        schema,
        definition=IndexDefinition(
            prefix=[PREFIX_TOOLS],
            index_type=IndexType.JSON,
        ),
    )
    logger.info("✅ Índice '%s' criado (Semantic Tool Routing).", IDX_TOOLS)


def inicializar_indices() -> None:
    """
    Ponto de entrada único para criação de índices.
    Chamado no startup do main.py antes da ingestão.

    ORDEM IMPORTA:
      1. chunks → precisa existir antes de ingerir PDFs
      2. tools  → precisa existir antes de registar as tools
    """
    try:
        criar_indice_chunks()
        criar_indice_tools()
        logger.info("✅ Todos os índices Redis prontos.")
    except Exception as e:
        logger.exception("❌ Falha ao criar índices Redis: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Operações de ingestão (usadas por rag/ingestor.py)
# ─────────────────────────────────────────────────────────────────────────────

def salvar_chunk(
    chunk_id: str,
    content: str,
    source: str,
    doc_type: str,
    embedding: list[float],
    chunk_index: int = 0,
    metadata: dict | None = None,
) -> None:
    """
    Salva um chunk de texto com embedding no Redis como JSON.

    FORMATO DA CHAVE: rag:chunk:{source}:{chunk_id}
    Exemplo: rag:chunk:edital_paes_2026.pdf:abc123

    O JSON armazenado:
      {
        "content":     "CURSO: Engenharia Civil | AC: 40 ...",
        "source":      "edital_paes_2026.pdf",
        "doc_type":    "edital",
        "chunk_index": 5,
        "embedding":   [0.123, -0.456, ...],    ← FLOAT32 como lista Python
        "metadata":    {"page": 3, "section": "Vagas"}
      }

    ECONOMIA DE RAM:
      - Stored como bytes serializado (não string Base64)
      - TTL=None → chunks são permanentes (re-ingestão limpa com DEL por prefix)
    """
    r = get_redis()
    key = f"{PREFIX_CHUNKS}{source}:{chunk_id}"

    doc = {
        "content":     content,
        "source":      source,
        "doc_type":    doc_type,
        "chunk_index": chunk_index,
        "embedding":   embedding,          # Lista de floats — RedisJSON serializa eficientemente
        "metadata":    metadata or {},
    }

    # JSON.SET armazena o documento inteiro; o índice actualiza-se automaticamente
    r.json().set(key, "$", doc)


def deletar_chunks_por_source(source: str) -> int:
    """
    Remove todos os chunks de um documento (útil na re-ingestão).
    Retorna o número de chaves deletadas.
    """
    r = get_redis()
    pattern = f"{PREFIX_CHUNKS}{source}:*"

    # SCAN iterativo é mais seguro que KEYS em produção (não bloqueia Redis)
    deleted = 0
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match=pattern, count=100)
        if keys:
            r.delete(*keys)
            deleted += len(keys)
        if cursor == 0:
            break

    logger.info("🗑️  Removidos %d chunks de '%s'", deleted, source)
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Busca Híbrida (usada por rag/hybrid_retriever.py)
# ─────────────────────────────────────────────────────────────────────────────

def busca_hibrida(
    query_text: str,
    query_embedding: list[float],
    source_filter: str | None = None,
    k_vector: int = 8,
    k_text: int = 8,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Busca híbrida: combina resultados vetoriais e BM25 via Reciprocal Rank Fusion.

    POR QUE RRF É MELHOR QUE NORMALIZAÇÃO DE SCORES?
    ─────────────────────────────────────────────────
      RRF = Σ 1/(k + rank_i)   onde k=60 (constante estabilizadora)

      Vantagens:
        - Não depende de escalas absolutas de score (vetor ≠ BM25)
        - Penaliza pouco a diferença entre posições altas (1 vs 2)
        - Penaliza muito documentos que aparecem só num dos métodos
        - Robusto a outliers

      Exemplo para datas exatas:
        Query: "matrícula veteranos 2026.1"
        Vetor: chunk semântico "período de matrícula semestral" → rank 1
        BM25:  chunk exato "EVENTO: Matrícula de veteranos | DATA: 03/02/2026" → rank 1
        RRF:   chunk exato sobe para o topo por aparecer nos dois métodos

    Parâmetros:
      query_text:      texto original para BM25
      query_embedding: vetor float32 para KNN
      source_filter:   filtro por documento (ex: "edital_paes_2026.pdf")
      k_vector:        top-K do resultado vetorial
      k_text:          top-K do resultado textual
      rrf_k:           constante de suavização do RRF (60 é o padrão da literatura)
    """
    import struct

    r = get_redis()

    # ── 1. Busca vetorial (KNN com HNSW) ─────────────────────────────────────
    # Serializa vetor como bytes FLOAT32 (exigido pelo RediSearch)
    embedding_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)

    # Monta filtro de source se fornecido
    # "@source:{nome_do_arquivo}" usa o TagField (match exato, case-insensitive)
    if source_filter:
        # Redis escapa hífens e pontos em tags com backslash
        safe_source = source_filter.replace(".", "\\.").replace("-", "\\-")
        pre_filter = f"(@source:{{{safe_source}}})"
        vec_query_str = f"{pre_filter}=>[KNN {k_vector} @embedding $vec AS vec_score]"
    else:
        vec_query_str = f"*=>[KNN {k_vector} @embedding $vec AS vec_score]"

    vec_query = (
        Query(vec_query_str)
        .sort_by("vec_score")
        .return_fields("content", "source", "doc_type", "chunk_index", "vec_score")
        .dialect(2)    # Dialeto 2 necessário para KNN queries
        .paging(0, k_vector)
    )

    try:
        vec_results = r.ft(IDX_CHUNKS).search(vec_query, {"vec": embedding_bytes})
        vec_docs = vec_results.docs
    except Exception as e:
        logger.warning("⚠️  Busca vetorial falhou: %s", e)
        vec_docs = []

    # ── 2. Busca por texto (BM25 + TF-IDF nativo do RediSearch) ─────────────
    # Escapa caracteres especiais para a sintaxe do RediSearch
    safe_text = _escapar_query_redis(query_text)

    if source_filter:
        safe_source = source_filter.replace(".", "\\.").replace("-", "\\-")
        txt_query_str = f"(@source:{{{safe_source}}}) ({safe_text})"
    else:
        txt_query_str = safe_text

    txt_query = (
        Query(txt_query_str)
        .return_fields("content", "source", "doc_type", "chunk_index")
        .paging(0, k_text)
    )

    try:
        txt_results = r.ft(IDX_CHUNKS).search(txt_query)
        txt_docs = txt_results.docs
    except Exception as e:
        logger.warning("⚠️  Busca textual falhou: %s", e)
        txt_docs = []

    # ── 3. Reciprocal Rank Fusion ────────────────────────────────────────────
    scores: dict[str, float] = {}

    for rank, doc in enumerate(vec_docs, start=1):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, doc in enumerate(txt_docs, start=1):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank)

    # ── 4. Merge e ordenação final ────────────────────────────────────────────
    # Reúne todos os docs únicos e ordena por score RRF decrescente
    all_docs: dict[str, Any] = {}
    for doc in vec_docs + txt_docs:
        if doc.id not in all_docs:
            all_docs[doc.id] = doc

    resultados = sorted(
        [
            {
                "id":          doc_id,
                "content":     getattr(all_docs[doc_id], "content", ""),
                "source":      getattr(all_docs[doc_id], "source", ""),
                "doc_type":    getattr(all_docs[doc_id], "doc_type", ""),
                "chunk_index": getattr(all_docs[doc_id], "chunk_index", 0),
                "rrf_score":   score,
            }
            for doc_id, score in scores.items()
            if doc_id in all_docs
        ],
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    logger.debug(
        "🔍 Busca híbrida | vetor=%d | texto=%d | merged=%d",
        len(vec_docs), len(txt_docs), len(resultados),
    )
    return resultados


def _escapar_query_redis(texto: str) -> str:
    """
    Escapa caracteres especiais da sintaxe RediSearch.
    Mantém os termos mais relevantes para BM25.

    RediSearch reserva: @ ! { } [ ] ( ) | - + * ? ~ ^
    """
    # Remove caracteres especiais mas mantém espaços e alfanuméricos
    import re
    # Preserva termos importantes como "2026.1", "BR-PPI"
    texto_limpo = re.sub(r'[!@\[\]{}()|~^]', ' ', texto)
    termos = texto_limpo.split()

    # Filtra termos muito curtos e palavras de parada
    stopwords_pt = {"de", "do", "da", "o", "a", "os", "as", "e", "em", "para",
                    "por", "com", "um", "uma", "que", "se", "no", "na", "nos", "nas"}
    termos_filtrados = [t for t in termos if len(t) > 2 and t.lower() not in stopwords_pt]

    if not termos_filtrados:
        return texto[:100]

    # Usa OR para termos → mais resultados na fusão
    return " | ".join(termos_filtrados[:10])   # Limita a 10 termos para performance


# ─────────────────────────────────────────────────────────────────────────────
# Gestão da memória de trabalho (Working Memory)
# ─────────────────────────────────────────────────────────────────────────────

def get_working_memory(session_id: str) -> dict:
    """
    Retorna a memória de trabalho da sessão atual.

    Working Memory contém:
      - ultimo_topico:     assunto da última pergunta
      - tool_usada:        última tool chamada
      - contexto_recuperado: trecho mais relevante encontrado
      - iteracoes:         contador de turns na conversa
    """
    r_text = get_redis_text()
    key = f"{PREFIX_WORKING_MEM}{session_id}"
    try:
        data = r_text.hgetall(key)
        return {k: v for k, v in data.items()} if data else {}
    except Exception:
        return {}


def set_working_memory(session_id: str, dados: dict, ttl: int = 1800) -> None:
    """Actualiza (merge) a memória de trabalho."""
    r_text = get_redis_text()
    key = f"{PREFIX_WORKING_MEM}{session_id}"
    try:
        if dados:
            r_text.hset(key, mapping=dados)
            r_text.expire(key, ttl)
    except Exception as e:
        logger.warning("⚠️  set_working_memory [%s]: %s", session_id, e)


def get_facts(user_id: str, limit: int = 10) -> list[str]:
    """
    Retorna os fatos extraídos sobre o utilizador da Long-Term Memory.

    Fatos são extraídos assincronamente pela rotina noturna (memory/extractor.py).
    Exemplos:
      - "Aluno do curso de Engenharia Civil"
      - "Inscrito no PAES 2026 categoria BR-PPI"
      - "Matrícula veterano realizada em fevereiro"
    """
    r_text = get_redis_text()
    key = f"{PREFIX_FACTS}{user_id}"
    try:
        facts = r_text.lrange(key, 0, limit - 1)
        return [f for f in facts if f]
    except Exception:
        return []


def add_fact(user_id: str, fact: str, ttl: int = 86400 * 30) -> None:
    """
    Adiciona um fato à Long-Term Memory do utilizador.
    TTL padrão: 30 dias (facts persistem entre semestres).
    """
    r_text = get_redis_text()
    key = f"{PREFIX_FACTS}{user_id}"
    try:
        r_text.lpush(key, fact)
        r_text.ltrim(key, 0, 49)    # Máximo 50 fatos por utilizador
        r_text.expire(key, ttl)
    except Exception as e:
        logger.warning("⚠️  add_fact [%s]: %s", user_id, e)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnóstico
# ─────────────────────────────────────────────────────────────────────────────

def diagnosticar() -> dict:
    """
    Retorna informações de saúde dos índices e memória.
    Chamado no startup (DEV_MODE) e no endpoint /banco/sources.
    """
    r = get_redis()
    r_text = get_redis_text()
    resultado = {}

    # Contagem de chunks por source
    try:
        cursor, keys = r.scan(0, match=f"{PREFIX_CHUNKS}*", count=1000)
        resultado["total_chunks"] = len(keys)

        sources: dict[str, int] = {}
        for key in keys:
            # Extrai o nome do source da chave: rag:chunk:{source}:{id}
            partes = key.decode().split(":", 3)
            if len(partes) >= 3:
                source = partes[2]
                sources[source] = sources.get(source, 0) + 1
        resultado["sources"] = sources
    except Exception as e:
        resultado["sources"] = {"erro": str(e)}

    # Info dos índices
    for idx_name in [IDX_CHUNKS, IDX_TOOLS]:
        try:
            info = r.ft(idx_name).info()
            resultado[idx_name] = {
                "num_docs":    info.get("num_docs", 0),
                "num_terms":   info.get("num_terms", 0),
                "indexing":    info.get("indexing", 0),
            }
        except Exception:
            resultado[idx_name] = {"status": "não existe"}

    # Memória Redis
    try:
        info_mem = r.info("memory")
        resultado["redis_ram_mb"] = round(
            info_mem.get("used_memory", 0) / 1024 / 1024, 2
        )
    except Exception:
        pass

    logger.info("🔍 Diagnóstico Redis: %s", resultado)
    return resultado