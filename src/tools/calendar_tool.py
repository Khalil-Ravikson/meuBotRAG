"""
tools/tool_calendario.py — Tool de Consulta ao Calendário Acadêmico
====================================================================
CORREÇÃO CRÍTICA vs versão anterior:
  SOURCE_CALENDARIO era "calendario_academico.pdf" (underscore)
  mas o arquivo real é "calendario-academico-2026.pdf" (hífen + ano).
  Isso causava o "Não encontrei" mesmo com o banco populado.

  ⚠️  Confirme o nome exato via Ingestor().diagnosticar() após a ingestão.
      O valor abaixo DEVE ser idêntico à chave em rag/ingestor.py:PDF_CONFIG.
"""
from __future__ import annotations
import unicodedata
import logging
from langchain_core.tools import tool
from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

MAX_CHARS = 1200

# ⚠️  Deve bater EXATAMENTE com a chave em rag/ingestor.py:PDF_CONFIG
SOURCE_CALENDARIO = "calendario-academico-2026.pdf"


def _normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return s.lower().strip()


def get_tool_calendario():
    """Fábrica: configura e retorna a @tool com retriever especializado."""
    vectorstore = get_vector_store()  # singleton — sem custo adicional

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 25,
            "lambda_mult": 0.75,   # 75% relevância, 25% diversidade
            "filter": {"source": SOURCE_CALENDARIO},
        },
    )

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

            docs = retriever.invoke(query_norm)

            if not docs:
                return (
                    "Não encontrei essa informação no calendário acadêmico. "
                    "Tente com outras palavras como: matrícula, feriado, prova, "
                    "trancamento, início das aulas, semestre."
                )

            for i, doc in enumerate(docs):
                logger.debug(
                    "📅 Chunk %d | source: %s | %s",
                    i + 1,
                    doc.metadata.get("source", "?"),
                    doc.page_content[:80].replace("\n", " "),
                )

            blocos = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
            resposta = "\n---\n".join(blocos)

            if len(resposta) > MAX_CHARS:
                resposta = resposta[:MAX_CHARS] + "\n[...resultado truncado]"

            return resposta

        except Exception as e:
            logger.exception("❌ Erro na tool de calendário: %s", e)
            return "ERRO TÉCNICO NA FERRAMENTA — não tente novamente nesta resposta."

    return consultar_calendario_academico