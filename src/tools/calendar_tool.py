from langchain.tools.retriever import create_retriever_tool
from src.services.db_service import get_vector_store

def get_calendar_tool():
    """
    Cria a ferramenta de busca no calendário acadêmico.
    A IA usará esta ferramenta quando precisar consultar datas, prazos ou regras.
    """
    
    # 1. Pega o banco de dados (que você já populou na ingestão)
    vectorstore = get_vector_store()
    
    # 2. Transforma em um 'buscado' (Retriever)
    retriever = vectorstore.as_retriever(
        search_type="mmr", # MMR garante diversidade (não traz 5 trechos iguais)
        search_kwargs={
            "k": 5,           # Traz 5 pedaços
            "fetch_k": 20,    # Analisa 20 antes de filtrar
            "lambda_mult": 0.7 
        }
    )

    # 3. Empacota como uma Ferramenta que a IA entende
    tool = create_retriever_tool(
        retriever,
        name="consultar_calendario_academico",
        description="""
        USE ESTA FERRAMENTA PARA QUALQUER PERGUNTA SOBRE O CALENDÁRIO ACADÊMICO.
        Útil para encontrar datas de provas, feriados, prazos de matrícula, 
        início e fim de períodos letivos e regras institucionais descritas no PDF.
        O input deve ser a pergunta completa ou palavras-chave sobre a data desejada.
        """
    )
    
    return tool