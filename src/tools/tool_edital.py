"""
tools/tool_edital.py — Tool de Consulta ao Edital PAES 2026 (v3)
=================================================================

O QUE MUDOU vs versão anterior:
─────────────────────────────────
  REMOVIDO:
    - from src.rag.vector_store import get_vector_store  ← pgvector eliminado
    - vectorstore.as_retriever(search_type="similarity") ← LangChain retriever

  ADICIONADO:
    - busca_hibrida() do redis_client com k_text maior
    - get_embeddings() apenas para o modelo CPU (não o store)

  NOTA SOBRE k_text > k_vector PARA O EDITAL:
    O edital tem muitas siglas e números exactos (AC, PcD, BR-PPI, número de vagas).
    Aumentar k_text garante que o BM25 encontra os termos exactos mesmo que
    o vetor traga chunks mais "genéricos". O RRF depois filtra o melhor.
"""
from __future__ import annotations

import unicodedata
import logging

from langchain_core.tools import tool

from src.infrastructure.redis_client import busca_hibrida
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

MAX_CHARS = 1400

# Deve bater EXACTAMENTE com a chave em rag/ingestion.py:PDF_CONFIG
SOURCE_EDITAL = "edital_paes_2026.pdf"


def _normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return s.lower().strip()


def get_tool_edital():
    """Fábrica: configura e retorna a @tool com busca híbrida no Redis."""
    embeddings_model = get_embeddings()

    @tool
    def consultar_edital_paes_2026(query: str) -> str:
        """
        Consulta regras, vagas, cotas e procedimentos do Edital PAES 2026 da UEMA.

        Use para perguntas sobre:
          - Categorias de vagas: AC, PcD, BR-PPI, BR-Q, BR-DC, IR-PPI, CFO-PP
          - Número de vagas por curso
          - Regras de inscrição e documentação exigida
          - Cronograma do processo seletivo
          - Cursos ofertados, turnos e campus
          - Procedimentos de heteroidentificação

        Parâmetro query: palavras-chave sobre o que deseja consultar.
        Exemplos:
          "vagas ampla concorrencia engenharia civil"
          "documentos necessarios inscricao"
          "cotas rede publica BR-PPI"
          "cronograma inscricoes datas"
        """
        try:
            query_norm = _normalizar(query)
            logger.debug("📋 Edital | query: '%s' → '%s'", query, query_norm)

            vetor = embeddings_model.embed_query(query_norm)

            # k_text=8: edital tem siglas exactas (AC, BR-PPI) → BM25 é crítico
            resultados = busca_hibrida(
                query_text=query_norm,
                query_embedding=vetor,
                source_filter=SOURCE_EDITAL,
                k_vector=4,
                k_text=8,
            )

            if not resultados:
                return (
                    "Não encontrei essa informação no edital do PAES 2026. "
                    "Tente com palavras como: vagas, cotas, inscrição, documentos, "
                    "cronograma, curso, AC, PcD, BR-PPI."
                )

            for i, r in enumerate(resultados):
                logger.debug(
                    "📋 Chunk %d | score=%.3f | %s",
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
            logger.exception("❌ Erro na tool de edital: %s", e)
            return "ERRO TÉCNICO NA FERRAMENTA — não tente novamente nesta resposta."

    return consultar_edital_paes_2026