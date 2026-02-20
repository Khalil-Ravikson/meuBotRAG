"""
================================================================================
tool_edital.py — Tool de Consulta ao Edital do Processo Seletivo (PAES 2026)
================================================================================

RESUMO:
  Consulta regras, vagas, categorias e procedimentos do Edital do processo
  seletivo da UEMA para 2026.

SOBRE O PDF DO EDITAL:
  O edital da UEMA (ex: Edital_57-2025-GR-UEMA-_PAES_2026_FINAL.pdf) tem:
    - Tabelas de vagas por curso com categorias (AC, PcD, BR-PPI, BR-Q, etc.)
    - Regras de inscrição e documentação
    - Cronograma do processo seletivo
    - Descrição das cotas e reservas de vagas
    - Informações sobre cursos, turnos e campus

  DESAFIO DESSE PDF:
    Tabelas com células mescladas (ex: "Reserva para candidatos da rede pública"
    abrange várias subcategorias) são difíceis para qualquer parser.
    O LlamaParse com parsing_instruction específica para editais universitários
    produz resultados melhores que o modo padrão.

  ESTRATÉGIA:
    - chunk_size menor (400) para não misturar regras de cotas diferentes
    - retriever com k=3 para trazer regras do contexto exato perguntado
    - Metadado source filtra só o edital

TOOLS COMENTADAS (para LLM superior no futuro):
  # - Busca de vagas por curso específico
  # - Comparação de cotas entre cursos
  # - Resposta livre sobre qualquer item do edital
================================================================================
"""

import unicodedata
import logging
from langchain_core.tools import tool
from src.services.db_service import get_vector_store

logger = logging.getLogger(__name__)

MAX_CHARS = 1400   # Edital tem parágrafos maiores, limite um pouco maior
SOURCE_EDITAL = "edital_paes_2026.pdf"


def _normalizar(texto: str) -> str:
    """Remove acentos e coloca em minúsculas para melhor matching."""
    sem_acento = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return sem_acento.lower().strip()


def get_tool_edital():
    """
    Fábrica da tool de edital.
    Configura e retorna a @tool com retriever filtrado no edital.
    """
    vectorstore = get_vector_store()

    # Para o edital usamos similarity (não MMR) porque as seções são bem distintas
    # e não precisamos de diversidade — queremos os chunks mais similares à query.
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
          - Categorias de vagas: Ampla Concorrência (AC), PcD, BR-PPI, BR-Q,
            BR-DC, IR-PPI, CFO-PP e demais cotas
          - Número de vagas por curso
          - Regras de inscrição e documentação exigida
          - Cronograma do processo seletivo (inscrições, resultados, matrículas)
          - Cursos ofertados, turnos e campus
          - Procedimentos de heteroidentificação

        Parâmetro query: palavras-chave sobre o que deseja consultar.
        Exemplos:
          "vagas ampla concorrencia engenharia civil"
          "documentos necessarios inscricao"
          "cotas rede publica BR-PPI"
          "cronograma inscricoes datas"
          "cursos campus paulo vi"
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
                    "📋 Chunk %d | source: %s | prévia: %s",
                    i + 1,
                    doc.metadata.get("source", "?"),
                    doc.page_content[:100].replace("\n", " "),
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