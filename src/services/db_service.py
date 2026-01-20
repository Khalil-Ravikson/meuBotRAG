# src/services/db_service.py
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings

def get_vector_store():
    """
    Fabrica e retorna a instância do banco vetorial conectado ao Postgres.
    """
    # 1. Define qual modelo de Embeddings vamos usar (O mesmo da ingestão)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 2. Conecta ao Postgres via Connection String do settings
    # collection_name é a "tabela" onde os vetores ficam
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="receitas_bot",
        connection=settings.DATABASE_URL,
        use_jsonb=True,
    )
    
    return vectorstore