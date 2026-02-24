"""
tools/tool_edital.py — Tool de Consulta ao Edital PAES 2026
============================================================
"""
from __future__ import annotations
import unicodedata
import logging
from langchain_core.tools import tool
from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

MAX_CHARS = 1400
# ⚠️  Deve bater EXATAMENTE com a chave em rag/ingestor.py:PDF_CONFIG
SOURCE_EDITAL = "edital_paes_2026.pdf"


def _normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return s.lower().strip()


def get_tool_edital():
    """Fábrica: configura e retorna a @tool com retriever filtrado no edital."""
    vectorstore = get_vector_store()  # singleton

    # Similarity (não MMR) para edital: as seções são distintas,
    # não precisamos de diversidade — queremos os chunks mais similares.
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "filter": {"source": SOURCE_EDITAL},
        },
    )

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

            docs = retriever.invoke(query_norm)

            if not docs:
                return (
                    "Não encontrei essa informação no edital do PAES 2026. "
                    "Tente com palavras como: vagas, cotas, inscrição, documentos, "
                    "cronograma, curso, AC, PcD, BR-PPI."
                )

            for i, doc in enumerate(docs):
                logger.debug(
                    "📋 Chunk %d | source: %s | %s",
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
            logger.exception("❌ Erro na tool de edital: %s", e)
            return "ERRO TÉCNICO NA FERRAMENTA — não tente novamente nesta resposta."

    return consultar_edital_paes_2026