"""
tools/tool_contatos.py — Tool de Consulta ao Guia de Contatos (v3)
===================================================================

O QUE MUDOU vs versão anterior:
─────────────────────────────────
  REMOVIDO:
    - from src.rag.vector_store import get_vector_store  ← pgvector eliminado
    - vectorstore.as_retriever(search_type="mmr", ...)   ← LangChain retriever

  ADICIONADO:
    - busca_hibrida() do redis_client com k_vector maior
    - get_embeddings() apenas para o modelo CPU

  NOTA SOBRE k_vector > k_text PARA CONTATOS:
    Contatos têm nomes de pessoas e setores que variam na escrita
    ("PROG" vs "Pró-Reitoria de Graduação" vs "pro-reitoria graduação").
    O vector capta estas variações semânticas melhor que o BM25.
    Mantemos k_text razoável para apanhar siglas exactas (CTIC, CECEN).
"""
from __future__ import annotations

import unicodedata
import logging

from langchain_core.tools import tool

from src.infrastructure.redis_client import busca_hibrida
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

MAX_CHARS = 1500

# Deve bater EXACTAMENTE com a chave em rag/ingestion.py:PDF_CONFIG
SOURCE_CONTATOS = "guia_contatos_2025.pdf"


def _normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return s.lower().strip()


def get_tool_contatos():
    """Fábrica: configura e retorna a @tool com busca híbrida no Redis."""
    embeddings_model = get_embeddings()

    @tool
    def consultar_contatos_uema(query: str) -> str:
        """
        Consulta e-mails, telefones e responsáveis de departamentos e setores da UEMA.

        Use para perguntas sobre:
          - E-mail ou telefone de uma pró-reitoria (PROG, PROEXAE, PRPPG, PRAD)
          - Contato de um centro acadêmico (CECEN, CESB, CESC, CCSA, etc.)
          - Coordenador ou diretor de um curso específico
          - Contato do CTIC (setor de TI)
          - Telefone ou e-mail da reitoria ou vice-reitoria
          - Secretaria acadêmica ou administrativa

        Parâmetro query: nome do setor, cargo ou curso que deseja o contato.
        Exemplos:
          "email PROG pro-reitoria graduacao"
          "telefone coordenador curso matematica"
          "email CTIC ti suporte"
          "contato reitoria vice-reitor"
        """
        try:
            query_norm = _normalizar(query)
            logger.debug("📞 Contatos | query: '%s' → '%s'", query, query_norm)

            vetor = embeddings_model.embed_query(query_norm)

            # k_vector=7: nomes de setores têm muitas variações → vector é melhor
            resultados = busca_hibrida(
                query_text=query_norm,
                query_embedding=vetor,
                source_filter=SOURCE_CONTATOS,
                k_vector=7,
                k_text=5,
            )

            if not resultados:
                return (
                    "Não encontrei esse contato no guia institucional. "
                    "Tente com o nome do setor, curso ou cargo. "
                    "Exemplos: PROG, CECEN, reitoria, CTIC, coordenador de física."
                )

            for i, r in enumerate(resultados):
                logger.debug(
                    "📞 Chunk %d | score=%.3f | %s",
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
            logger.exception("❌ Erro na tool de contatos: %s", e)
            return "ERRO TÉCNICO NA FERRAMENTA — não tente novamente nesta resposta."

    return consultar_contatos_uema