"""
tools/tool_contatos.py — Tool de Consulta ao Guia de Contatos
=============================================================
"""
from __future__ import annotations
import unicodedata
import logging
from langchain_core.tools import tool
from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

MAX_CHARS = 1500
# ⚠️  Deve bater EXATAMENTE com a chave em rag/ingestor.py:PDF_CONFIG
SOURCE_CONTATOS = "guia_contatos_2025.pdf"


def _normalizar(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return s.lower().strip()


def get_tool_contatos():
    """Fábrica: usa MMR para trazer contatos variados (evita repetição)."""
    vectorstore = get_vector_store()  # singleton

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.65,   # mais diversidade: contatos de setores diferentes
            "filter": {"source": SOURCE_CONTATOS},
        },
    )

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

            docs = retriever.invoke(query_norm)

            if not docs:
                return (
                    "Não encontrei esse contato no guia institucional. "
                    "Tente com o nome do setor, curso ou cargo. "
                    "Exemplos: PROG, CECEN, reitoria, CTIC, coordenador de física."
                )

            for i, doc in enumerate(docs):
                logger.debug(
                    "📞 Chunk %d | source: %s | %s",
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
            logger.exception("❌ Erro na tool de contatos: %s", e)
            return "ERRO TÉCNICO NA FERRAMENTA — não tente novamente nesta resposta."

    return consultar_contatos_uema