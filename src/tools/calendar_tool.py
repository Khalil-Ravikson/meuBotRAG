"""
calendar_tool.py — Tool de calendário com resposta compacta

O problema: o retriever devolvia chunks enormes (tabelas inteiras do PDF),
estourando o contexto do LLaMA e causando o erro de tool_use_failed na
chamada SEGUINTE (o modelo "esquece" como formatar JSON depois de processar
muito texto).

Solução: wrapper que trunca a resposta da tool para no máximo MAX_CHARS,
garantindo que o modelo sempre tenha espaço para raciocinar depois.
"""

from langchain_core.tools import tool
from src.services.db_service import get_vector_store

# Limite seguro: ~800 chars cobre 2-3 entradas do calendário sem estourar contexto
MAX_CHARS = 900


def get_calendar_tool():
    vectorstore = get_vector_store()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 15,
            "lambda_mult": 0.7,
        },
    )

    @tool
    def consultar_calendario_academico(query: str) -> str:
        """
        Consulta datas, prazos e períodos do calendário acadêmico da UEMA.
        Use para: matrículas, provas, feriados, início/fim de semestres, trancamentos.
        Parâmetro: query — palavras-chave da data ou evento que deseja consultar.
        """
        try:
            docs = retriever.invoke(query)

            if not docs:
                return "Nenhuma informação encontrada no calendário para essa consulta."

            # Concatena apenas o conteúdo relevante
            conteudo = "\n\n".join(
                doc.page_content.strip()
                for doc in docs
                if doc.page_content.strip()
            )

            # Trunca para evitar estouro de contexto no LLaMA
            if len(conteudo) > MAX_CHARS:
                conteudo = conteudo[:MAX_CHARS] + "\n[...resultado truncado para brevidade]"

            return conteudo

        except Exception as e:
            return f"Erro ao consultar calendário: {str(e)}"

    return consultar_calendario_academico