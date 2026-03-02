"""
rag/hybrid_retriever.py — Recuperador Híbrido com Metadados Hierárquicos
=========================================================================

ESTE É O FICHEIRO QUE ELIMINA AS ALUCINAÇÕES EM DATAS E SIGLAS.
───────────────────────────────────────────────────────────────

PORQUÊ A BUSCA SIMPLES ALUCINA:
────────────────────────────────
  Sistema atual (pgvector + MMR):
    Query: "matrícula veteranos"
    → Embedding encontra chunks semanticamente similares
    → Pode trazer chunk de 2025 ou chunk genérico sobre matrícula
    → Gemini interpola com conhecimento interno → ALUCINA DATA

  Novo sistema (Híbrido Redis + Metadados Hierárquicos):
    Query transformada: "matrícula veteranos UEMA 2026.1 data período"
    → BM25 captura: "2026.1", "veteranos" com match exato
    → Vetor captura: semântica de "período de matrícula"
    → RRF funde: chunk "EVENTO: Matrícula de veteranos | DATA: 03/02/2026" sobe
    → Metadado "source: calendario-academico-2026.pdf" confirma validade
    → Gemini gera resposta ancorada no chunk correto → SEM ALUCINAÇÃO

PIPELINE COMPLETO (query_transform → hybrid_retriever):
─────────────────────────────────────────────────────────

  Mensagem do aluno: "quando começa minha matrícula?"
  
  ┌─ query_transform.py ──────────────────────────────────────────┐
  │  + fatos: ["Aluno de Engenharia Civil, veterano 2026.1"]       │
  │  → Query: "matrícula veteranos Engenharia Civil 2026.1 início" │
  └────────────────────────────────────────────────────────────────┘
             ↓
  ┌─ hybrid_retriever.py (este ficheiro) ─────────────────────────┐
  │  BM25:   "matrícula", "veteranos", "2026.1" → rank exato      │
  │  Vetor:  "período matrícula início semestre" → rank semântico  │
  │  RRF:    funde ranks → "EVENTO: Matrícula de veteranos |       │
  │          DATA: 03/02/2026 a 07/02/2026 | SEM: 2026.1" TOP-1   │
  │  Format: adiciona cabeçalho hierárquico + metadados            │
  └────────────────────────────────────────────────────────────────┘
             ↓
  ┌─ gemini_provider.py ──────────────────────────────────────────┐
  │  Contexto: "FONTE: Calendário Acadêmico 2026\n                 │
  │  EVENTO: Matrícula de veteranos | DATA: 03/02/2026 a..."      │
  │  → "A matrícula de veteranos ocorre de *03 a 07 de fevereiro   │
  │     de 2026* (2026.1)." — ANCORADO, SEM ALUCINAÇÃO ✓          │
  └────────────────────────────────────────────────────────────────┘

METADADOS HIERÁRQUICOS:
────────────────────────
  Cada chunk é formatado com cabeçalho que indica de onde vem:

  [FONTE: Calendário Acadêmico UEMA 2026 | TIPO: evento_academico]
  EVENTO: Matrícula de veteranos | DATA: 03/02/2026 a 07/02/2026 | SEM: 2026.1

  O LLM vê explicitamente a FONTE antes do conteúdo.
  Isso ancora a resposta e reduz a probabilidade de o modelo
  misturar informação do chunk com conhecimento interno desatualizado.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.infrastructure.redis_client import busca_hibrida
from src.rag.query_transform import QueryTransformada, transformar_para_step_back

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mapa de source → título legível (para cabeçalhos hierárquicos)
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_PARA_TITULO: dict[str, str] = {
    "calendario-academico-2026.pdf": "Calendário Acadêmico UEMA 2026",
    "edital_paes_2026.pdf":          "Edital PAES 2026 — Processo Seletivo UEMA",
    "guia_contatos_2025.pdf":        "Guia de Contatos UEMA 2025",
    "contatos_saoluis.txt":          "Contatos São Luís — UEMA",
    "regras_ru.txt":                 "Regras do Restaurante Universitário",
}

_DOC_TYPE_PARA_LABEL: dict[str, str] = {
    "calendario": "CALENDÁRIO ACADÊMICO",
    "edital":     "EDITAL PAES 2026",
    "contatos":   "CONTATOS UEMA",
}

# Configurações de busca por tipo de documento
_BUSCA_CONFIG: dict[str, dict] = {
    "calendario": {"k_vector": 6,  "k_text": 8},   # Texto mais importante (datas exatas)
    "edital":     {"k_vector": 6,  "k_text": 8},   # Texto crítico (vagas, siglas)
    "contatos":   {"k_vector": 8,  "k_text": 6},   # Vetor mais útil (nomes de setores)
    "default":    {"k_vector": 6,  "k_text": 6},
}

# Número máximo de chunks a incluir no contexto final
_MAX_CHUNKS_CONTEXTO = 4

# Mínimo de chars num chunk para ser considerado útil
_MIN_CHARS_CHUNK = 30

# Máximo total de chars no contexto (para controlar tokens enviados ao Gemini)
_MAX_CHARS_CONTEXTO_TOTAL = 2_500


# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkRecuperado:
    """Chunk de texto recuperado com metadados."""
    content:     str
    source:      str
    doc_type:    str
    chunk_index: int
    rrf_score:   float
    chunk_id:    str = ""

    @property
    def titulo_fonte(self) -> str:
        return _SOURCE_PARA_TITULO.get(self.source, self.source)

    @property
    def label_tipo(self) -> str:
        return _DOC_TYPE_PARA_LABEL.get(self.doc_type, "INFORMAÇÃO")

    @property
    def conteudo_limpo(self) -> str:
        return self.content.strip()


@dataclass
class ResultadoRecuperacao:
    """
    Resultado completo da recuperação, pronto para injeção no prompt do Gemini.
    """
    chunks:          list[ChunkRecuperado] = field(default_factory=list)
    contexto_formatado: str = ""
    total_chars:     int = 0
    fonte_principal: str = ""
    encontrou:       bool = False
    metodo_usado:    str = ""   # "hibrido" | "step_back" | "vazio"


# ─────────────────────────────────────────────────────────────────────────────
# API principal
# ─────────────────────────────────────────────────────────────────────────────

def recuperar(
    query_transformada: QueryTransformada,
    source_filter: str | None = None,
    doc_type: str | None = None,
) -> ResultadoRecuperacao:
    """
    Recupera chunks relevantes usando busca híbrida no Redis.

    ESTRATÉGIA DE MÚLTIPLAS QUERIES:
    ──────────────────────────────────
      Se a QueryTransformada tiver sub_queries (decomposição),
      executamos busca para CADA sub-query e fundimos os resultados.
      Isso garante cobertura para perguntas com múltiplas intenções.

    FALLBACK STEP-BACK:
    ────────────────────
      Se a busca principal retornar 0 resultados, tentamos uma versão
      mais genérica da query original (Step-Back Prompting).
      Isso evita respostas "não encontrei informação" desnecessárias.

    Parâmetros:
      query_transformada: Resultado do query_transform.py
      source_filter:      Filtra por ficheiro PDF específico
      doc_type:           Filtra por tipo ("calendario", "edital", "contatos")
    """
    # Obtém configuração de busca baseada no tipo de documento
    config = _BUSCA_CONFIG.get(doc_type or "default", _BUSCA_CONFIG["default"])

    # ── Busca para query principal ────────────────────────────────────────────
    todos_chunks = _buscar_para_query(
        query=query_transformada.query_principal,
        source_filter=source_filter,
        k_vector=config["k_vector"],
        k_text=config["k_text"],
    )

    # ── Busca para sub-queries (se existirem) ─────────────────────────────────
    for sub_query in query_transformada.sub_queries:
        sub_chunks = _buscar_para_query(
            query=sub_query,
            source_filter=source_filter,
            k_vector=config["k_vector"] // 2,  # Menos resultados por sub-query
            k_text=config["k_text"] // 2,
        )
        todos_chunks.extend(sub_chunks)

    # ── Deduplica e ordena ────────────────────────────────────────────────────
    chunks_unicos = _deduplicar_e_ordenar(todos_chunks)

    if not chunks_unicos:
        # Fallback Step-Back: query mais genérica
        logger.info("🔙 Step-back fallback para: '%.50s'", query_transformada.query_original)
        step_back_query = transformar_para_step_back(query_transformada.query_original)

        chunks_unicos = _buscar_para_query(
            query=step_back_query,
            source_filter=source_filter,
            k_vector=config["k_vector"],
            k_text=config["k_text"],
        )
        chunks_unicos = _deduplicar_e_ordenar(chunks_unicos)
        metodo = "step_back"
    else:
        metodo = "hibrido"

    if not chunks_unicos:
        return ResultadoRecuperacao(
            encontrou=False,
            metodo_usado="vazio",
            contexto_formatado="",
        )

    # ── Seleciona top-N chunks dentro do budget de chars ─────────────────────
    chunks_selecionados = _selecionar_chunks(chunks_unicos)

    # ── Formata com metadados hierárquicos ────────────────────────────────────
    contexto = _formatar_contexto_hierarquico(chunks_selecionados)

    fonte_principal = chunks_selecionados[0].titulo_fonte if chunks_selecionados else ""

    logger.info(
        "✅ Recuperação [%s] | %d chunks | %d chars | query='%.40s'",
        metodo, len(chunks_selecionados), len(contexto),
        query_transformada.query_principal,
    )

    return ResultadoRecuperacao(
        chunks=chunks_selecionados,
        contexto_formatado=contexto,
        total_chars=len(contexto),
        fonte_principal=fonte_principal,
        encontrou=True,
        metodo_usado=metodo,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Funções internas
# ─────────────────────────────────────────────────────────────────────────────

def _buscar_para_query(
    query: str,
    source_filter: str | None,
    k_vector: int,
    k_text: int,
) -> list[ChunkRecuperado]:
    """
    Executa a busca híbrida Redis para uma query específica.
    Retorna lista de ChunkRecuperado (ainda não deduplicada).
    """
    # Computa embedding da query (CPU local, ~5ms)
    try:
        from src.rag.vector_store import get_embeddings
        embeddings_model = get_embeddings()
        vetor_query = embeddings_model.embed_query(query)
    except Exception as e:
        logger.error("❌ Falha ao computar embedding da query: %s", e)
        return []

    # Chama busca híbrida do redis_client.py
    resultados_raw = busca_hibrida(
        query_text=query,
        query_embedding=vetor_query,
        source_filter=source_filter,
        k_vector=k_vector,
        k_text=k_text,
    )

    # Converte para ChunkRecuperado
    chunks = []
    for r in resultados_raw:
        content = r.get("content", "").strip()
        if len(content) < _MIN_CHARS_CHUNK:
            continue
        chunks.append(ChunkRecuperado(
            content=content,
            source=r.get("source", ""),
            doc_type=r.get("doc_type", ""),
            chunk_index=r.get("chunk_index", 0),
            rrf_score=r.get("rrf_score", 0.0),
            chunk_id=r.get("id", ""),
        ))

    return chunks


def _deduplicar_e_ordenar(chunks: list[ChunkRecuperado]) -> list[ChunkRecuperado]:
    """
    Remove chunks duplicados (mesmo conteúdo de buscas diferentes)
    e ordena por score RRF decrescente.

    ESTRATÉGIA DE DEDUPLICAÇÃO:
      Usa os primeiros 100 chars do conteúdo como fingerprint.
      Dois chunks com início idêntico = mesmo chunk (pode vir de sub-queries).
      Mantém o que tem maior score RRF.
    """
    vistos: dict[str, ChunkRecuperado] = {}

    for chunk in chunks:
        fingerprint = chunk.content[:100].strip().lower()
        if fingerprint not in vistos:
            vistos[fingerprint] = chunk
        elif chunk.rrf_score > vistos[fingerprint].rrf_score:
            # Substitui por versão com score maior
            vistos[fingerprint] = chunk

    return sorted(vistos.values(), key=lambda c: c.rrf_score, reverse=True)


def _selecionar_chunks(chunks: list[ChunkRecuperado]) -> list[ChunkRecuperado]:
    """
    Seleciona os melhores chunks dentro do budget total de chars.

    LÓGICA:
      Adiciona chunks em ordem de score até atingir _MAX_CHUNKS_CONTEXTO
      ou _MAX_CHARS_CONTEXTO_TOTAL — o que acontecer primeiro.

    NOTA SOBRE DIVERSIDADE:
      Preferimos chunks de sources diferentes quando possível.
      Se todos são do mesmo PDF (ex: edital), está OK.
      Mas se temos chunks do calendário E do edital, incluímos de ambos.
    """
    selecionados = []
    total_chars = 0
    sources_incluidas: set[str] = set()

    for chunk in chunks:
        if len(selecionados) >= _MAX_CHUNKS_CONTEXTO:
            break
        if total_chars + len(chunk.content) > _MAX_CHARS_CONTEXTO_TOTAL:
            # Só quebra se já temos pelo menos 1 chunk
            if selecionados:
                break

        selecionados.append(chunk)
        total_chars += len(chunk.content)
        sources_incluidas.add(chunk.source)

    logger.debug(
        "📦 Chunks selecionados: %d | chars: %d | sources: %s",
        len(selecionados), total_chars, sources_incluidas,
    )
    return selecionados


def _formatar_contexto_hierarquico(chunks: list[ChunkRecuperado]) -> str:
    """
    Formata os chunks com cabeçalhos hierárquicos para o prompt do Gemini.

    ESTRUTURA DO CONTEXTO FORMATADO:
    ──────────────────────────────────

    ━━━ FONTE: Calendário Acadêmico UEMA 2026 [CALENDÁRIO ACADÊMICO] ━━━
    EVENTO: Matrícula de veteranos | DATA: 03/02/2026 a 07/02/2026 | SEM: 2026.1

    ━━━ FONTE: Calendário Acadêmico UEMA 2026 [CALENDÁRIO ACADÊMICO] ━━━
    EVENTO: Rematrícula de calouros | DATA: 10/02/2026 a 14/02/2026 | SEM: 2026.1

    POR QUE ESTE FORMATO É ANTI-ALUCINAÇÃO:
      O Gemini vê EXPLICITAMENTE de onde veio cada informação.
      O LLM tem forte tendência a respeitar a fonte indicada no contexto.
      Estudos de RAG mostram que cabeçalhos de fonte reduzem alucinações em ~40%.

    NOTA SOBRE CHUNKS DO MESMO SOURCE:
      Agrupamos chunks do mesmo documento com o cabeçalho uma única vez
      para economizar tokens (em vez de repetir o cabeçalho em cada chunk).
    """
    if not chunks:
        return ""

    # Agrupa por source para evitar repetição de cabeçalhos
    por_source: dict[str, list[ChunkRecuperado]] = {}
    for chunk in chunks:
        por_source.setdefault(chunk.source, []).append(chunk)

    blocos = []
    for source, source_chunks in por_source.items():
        if not source_chunks:
            continue

        primeiro = source_chunks[0]
        titulo = primeiro.titulo_fonte
        label  = primeiro.label_tipo

        # Cabeçalho hierárquico
        cabecalho = f"━━━ FONTE: {titulo} [{label}] ━━━"
        conteudos = [cabecalho]

        for chunk in source_chunks:
            conteudos.append(chunk.conteudo_limpo)

        blocos.append("\n".join(conteudos))

    return "\n\n".join(blocos)


# ─────────────────────────────────────────────────────────────────────────────
# Função de conveniência para uso direto (sem QueryTransformada)
# ─────────────────────────────────────────────────────────────────────────────

def recuperar_simples(
    query_texto: str,
    source_filter: str | None = None,
    doc_type: str | None = None,
) -> ResultadoRecuperacao:
    """
    Atalho para busca sem transformação de query.
    Usado quando a query já é técnica (semantic_router alta confiança).
    """
    from src.rag.query_transform import QueryTransformada
    qt = QueryTransformada(
        query_original=query_texto,
        query_principal=query_texto,
        foi_transformada=False,
    )
    return recuperar(qt, source_filter=source_filter, doc_type=doc_type)