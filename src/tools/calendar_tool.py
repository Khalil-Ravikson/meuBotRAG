"""
tools/calendar_tool.py — Tool de Consulta ao Calendário Acadêmico (v3)
=======================================================================

O QUE MUDOU vs versão anterior:
─────────────────────────────────
  REMOVIDO:
    - from src.rag.vector_store import get_vector_store   ← pgvector eliminado
    - vectorstore.as_retriever(search_type="mmr", ...)    ← LangChain retriever
    - filter por metadata do pgvector

  ADICIONADO:
    - from src.infrastructure.redis_client import busca_hibrida
    - from src.rag.vector_store import get_embeddings  (só o modelo CPU, não o store)
    - source_filter="calendario-academico-2026.pdf"    (mesmo valor de antes)

  MANTIDO:
    - SOURCE_CALENDARIO (deve bater com PDF_CONFIG em rag/ingestion.py)
    - Descrição da @tool (usada pelo semantic_router para embedding)
    - Interface de retorno (string) — compatível com o AgentCore v3
    - _normalizar() e MAX_CHARS

POR QUÊ A BUSCA HÍBRIDA É MELHOR QUE O MMR ANTERIOR:
───────────────────────────────────────────────────────
  MMR (Maximal Marginal Relevance) apenas diversificava os resultados vetoriais.
  Busca Híbrida (BM25 + Vector via RRF) captura ADICIONALMENTE:
    - Datas exactas: "03/02/2026" → BM25 match exacto, não apenas semântico
    - Siglas exactas: "2026.1", "substitutiva" → BM25 nunca confunde
    - Semântica: "início das aulas" ≈ "começo do semestre" → Vector capta
  Resultado: zero alucinações em datas quando o chunk existe no Redis.
"""
from __future__ import annotations

import unicodedata
import logging

from langchain_core.tools import tool

from src.infrastructure.redis_client import busca_hibrida
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

MAX_CHARS = 1200

# Deve bater EXACTAMENTE com a chave em rag/ingestion.py:PDF_CONFIG
SOURCE_CALENDARIO = "calendario-academico-2026.pdf"


def _normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return s.lower().strip()


def get_tool_calendario():
    """
    Fábrica: configura e retorna a @tool com busca híbrida no Redis.
    Sem estado interno — a busca é feita em cada chamada.
    """
    embeddings_model = get_embeddings()  # singleton — carregado uma vez

    @tool
    def consultar_calendario_academico(query: str) -> str:
        """
        Consulta datas, prazos e eventos do calendário acadêmico da UEMA 2026.

        Use para perguntas sobre:
          - Matrícula e rematrícula (veteranos, calouros, retardatários, reingressos)
          - Início e fim de semestres letivos (2026.1 e 2026.2)
          - Feriados e recessos acadêmicos
          - Provas, avaliações finais e substitutivas
          - Trancamento de matrícula ou de curso
          - Defesas, bancas, prazos de entrega

        Parâmetro query: palavras-chave do evento desejado.
        Exemplos:
          "matricula veteranos 2026.1"
          "feriados junho julho"
          "inicio aulas segundo semestre"
          "prazo trancamento"
        """
        try:
            query_norm = _normalizar(query)
            logger.debug("📅 Calendário | query: '%s' → '%s'", query, query_norm)

            # Gera embedding da query (CPU local, ~5ms)
            vetor = embeddings_model.embed_query(query_norm)

            # Busca híbrida: BM25 + Vector filtrado por source
            resultados = busca_hibrida(
                query_text=query_norm,
                query_embedding=vetor,
                source_filter=SOURCE_CALENDARIO,
                k_vector=5,
                k_text=6,
            )

            if not resultados:
                return (
                    "Não encontrei essa informação no calendário acadêmico. "
                    "Tente com outras palavras como: matrícula, feriado, prova, "
                    "trancamento, início das aulas, semestre."
                )

            for i, r in enumerate(resultados):
                logger.debug(
                    "📅 Chunk %d | score=%.3f | %s",
                    i + 1,
                    r.get("rrf_score", 0),
                    r.get("content", "")[:80].replace("\n", " "),
                )

            blocos   = [r["content"].strip() for r in resultados if r.get("content", "").strip()]
            resposta = "\n---\n".join(blocos)

            if len(resposta) > MAX_CHARS:
                resposta = resposta[:MAX_CHARS] + "\n[...resultado truncado]"

            return resposta

        except Exception as e:
            logger.exception("❌ Erro na tool de calendário: %s", e)
            return "ERRO TÉCNICO NA FERRAMENTA — não tente novamente nesta resposta."

    return consultar_calendario_academico