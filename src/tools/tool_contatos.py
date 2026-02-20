"""
================================================================================
tool_contatos.py — Tool de Consulta ao Guia de Contatos Institucional
================================================================================

RESUMO:
  Consulta e-mails, telefones e responsáveis de departamentos, cursos e
  pró-reitorias da UEMA a partir do PDF do Guia de Contatos 2025.

SOBRE O PDF DE CONTATOS:
  O "Guia-de-Contatos_final2.pdf" da UEMA tem:
    - Tabelas coloridas por centro/unidade (CECEN, CESB, CESC, etc.)
    - Colunas: CARGO/FUNÇÃO | GESTOR | E-MAIL | TELEFONE
    - Separação por campus e pró-reitorias
    - Cabeçalhos visuais com logos dos centros

  DESAFIO DESSE PDF:
    Tabelas com células coloridas (fundo azul/verde) podem confundir o parser
    ao extrair alinhamento de colunas. O LlamaParse no modo markdown tende a
    achatar essas tabelas, às vezes misturando e-mail com nome do gestor.

  ESTRATÉGIA:
    - parsing_instruction específica (configurada no rag_service) orienta
      o LlamaParse a tratar cada linha como: CARGO | NOME | EMAIL | TELEFONE
    - Cada contato é salvo como um chunk único e atômico na ingestão
    - retriever com k=4 traz múltiplos contatos quando a pergunta é ampla
      (ex: "contatos do CECEN" pode retornar vários coordenadores)

TOOLS COMENTADAS (para LLM superior no futuro):
  # - Listar todos os contatos de um centro específico
  # - Buscar por nome de pessoa (não só por cargo/função)
  # - Filtrar por campus (Paulo VI, Caxias, Imperatriz, etc.)
================================================================================
"""

import unicodedata
import logging
from langchain_core.tools import tool
from src.services.db_service import get_vector_store

logger = logging.getLogger(__name__)

MAX_CHARS = 1500   # Contatos precisam de mais espaço (e-mail + telefone + nome)
SOURCE_CONTATOS = "guia_contatos_2025.pdf"


def _normalizar(texto: str) -> str:
    """Remove acentos e coloca em minúsculas para matching robusto."""
    sem_acento = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return sem_acento.lower().strip()


def get_tool_contatos():
    """
    Fábrica da tool de contatos institucionais.
    Usa MMR para evitar retornar o mesmo departamento várias vezes.
    """
    vectorstore = get_vector_store()

    # MMR é importante aqui: quando alguém pergunta "contatos da pró-reitoria"
    # queremos pró-reitorias DIFERENTES, não o mesmo chunk repetido.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.65,  # mais diversidade para trazer contatos variados
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
          "contato diretor CECEN campus paulo vi"
          "telefone coordenador curso matematica"
          "email CTIC ti suporte"
          "contato reitoria"
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
                    "📞 Chunk %d | source: %s | prévia: %s",
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
            logger.exception("❌ Erro na tool de contatos: %s", e)
            return "ERRO TÉCNICO NA FERRAMENTA — não tente novamente nesta resposta."

    return consultar_contatos_uema