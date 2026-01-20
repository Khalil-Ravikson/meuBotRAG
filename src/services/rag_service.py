# src/services/rag_service.py
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.services.db_service import get_vector_store # <--- IMPORTA O DB

class RagService:
    def __init__(self):
        self.chain = None
        # Pega a conexão pronta do db_service
        self.vectorstore = get_vector_store()

    def inicializar(self):
        """Monta a corrente (Chain) de pensamento"""
        print("🧠 Inicializando RAG...")

        # LLM
        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY, 
            model="llama-3.3-70b-versatile", 
            temperature=0.3
        )
        
        # Prompt
        template = """
        Você é um assistente culinário útil.
        Use APENAS o contexto abaixo para responder. Se não souber, diga "Não encontrei no meu banco de dados".
        
        Contexto: {context}
        Pergunta: {input}
        """
        prompt = PromptTemplate.from_template(template)
        
        # Retriever (Buscador)
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

        # Chain
        self.chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("✅ RAG Pronto!")

    def ingerir_pdf(self):
        """Lê PDF e salva no Banco via db_service"""
        if not os.path.exists(settings.PDF_PATH):
            print(f"⚠️ PDF não encontrado: {settings.PDF_PATH}")
            return

        print("📚 Lendo PDF e inserindo no Postgres...")
        loader = PyMuPDFLoader(settings.PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        # Salva usando a conexão do db_service
        self.vectorstore.add_documents(chunks)
        print(f"✅ {len(chunks)} vetores salvos no Postgres!")

    def responder(self, pergunta: str):
        if not self.chain:
            return "Erro: O cérebro do robô ainda está carregando..."
        return self.chain.invoke(pergunta)
    