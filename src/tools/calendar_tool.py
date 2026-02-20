"""
================================================================================
tool_calendario.py — Tool de Consulta ao Calendário Acadêmico
================================================================================

RESUMO:
  Consulta datas, prazos e eventos do calendário acadêmico da UEMA 2026.
  Usa retriever filtrado EXCLUSIVAMENTE no PDF do calendário.

  Por que filtrar por source:
    Sem filtro, o retriever pode trazer chunks do edital ou de contatos
    quando a pergunta menciona palavras como "data" ou "prazo".

SOBRE O PDF DO CALENDÁRIO:
  PDFs de calendário da UEMA costumam ter tabelas mensais com:
    - Coluna de datas (dia/mês)
    - Coluna de eventos (ex: "Início das aulas", "Feriado estadual")
    - Coluna de semestre (2026.1 / 2026.2)

  O LlamaParse com result_type="markdown" converte bem essas tabelas simples.
  A pré-formatação em rag_service.py transforma cada linha em:
    "EVENTO: Início das aulas | DATA: 10/02/2026 | SEM: 2026.1"
  Isso melhora muito a precisão do embedding.

TOOLS COMENTADAS (para implementação futura com LLM superior):
  - Resposta livre a qualquer pergunta
  - Busca por múltiplos semestres simultaneamente
================================================================================
"""

import unicodedata
import logging
from langchain_core.tools import tool
from src.services.db_service import get_vector_store

logger = logging.getLogger(__name__)

# Limite de caracteres na resposta.
# ~1200 chars cobre 3-4 eventos do calendário com datas completas.
MAX_CHARS = 1200

# Nome exato do arquivo PDF do calendário (deve coincidir com o metadado 'source')
SOURCE_CALENDARIO = "calendario_academico.pdf"


def _normalizar(texto: str) -> str:
    """
    Remove acentos e coloca em minúsculas.
    Garante que "matrícula" == "matricula" no matching do retriever.
    """
    sem_acento = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return sem_acento.lower().strip()


def get_tool_calendario():
    """
    Fábrica da tool de calendário acadêmico.
    Configura e retorna a @tool com retriever especializado.
    """
    vectorstore = get_vector_store()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,           # retorna até 4 chunks mais relevantes
            "fetch_k": 25,    # avalia 25 candidatos antes de selecionar os 4
            "lambda_mult": 0.75,  # 75% relevância, 25% diversidade
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

            # Log de debug: mostra os chunks encontrados
            for i, doc in enumerate(docs):
                logger.debug(
                    "📅 Chunk %d | source: %s | prévia: %s",
                    i + 1,
                    doc.metadata.get("source", "?"),
                    doc.page_content[:100].replace("\n", " "),
                )

            # Monta resposta com separador claro entre chunks
            blocos = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
            resposta = "\n---\n".join(blocos)

            # Trunca se necessário (segurança para não estourar contexto)
            if len(resposta) > MAX_CHARS:
                resposta = resposta[:MAX_CHARS] + "\n[...resultado truncado]"

            return resposta

        except Exception as e:
            logger.exception("❌ Erro na tool de calendário: %s", e)
            return "ERRO TÉCNICO NA FERRAMENTA — não tente novamente nesta resposta."

    return consultar_calendario_academico