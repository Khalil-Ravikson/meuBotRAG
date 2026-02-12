import os
import nest_asyncio
import re
import glob

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import create_retriever_tool, StructuredTool
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# --- LlamaParse & Groq ---
from llama_parse import LlamaParse
from groq import Groq 

# --- Ferramentas Personalizadas ---
from src.tools.calendar_tool import get_calendar_tool  # Sua tool modular
from src.tools.email_tool import enviar_email_notificacao
from src.tools.fila import consultar_fila
from src.tools.glpi import abrir_chamado_glpi

# --- Serviços Internos ---
from src.services.logger_service import LogService
from src.config import settings
from src.services.db_service import get_vector_store
from src.services.redis_history import get_session_history

# Correção para loops de evento
nest_asyncio.apply()

# Instância global do Logger para uso na função de erro
logger = LogService()

# --- FUNÇÃO DE ERRO PERSONALIZADA (Auto-Cura) ---
def handle_tool_error(error: Exception) -> str:
    """Captura falhas nas tools, loga no Redis e orienta a IA a ser resiliente."""
    error_msg = str(error)
    logger.log_error("SYSTEM", "Tool Execution Failure", error_msg)
    
    return (
        f"ERRO TÉCNICO NA FERRAMENTA: {error_msg}. "
        "INSTRUÇÃO: Não tente usar esta ferramenta novamente com os mesmos termos. "
        "Informe ao usuário que o sistema de consulta está instável e peça para tentar mais tarde."
    )

class RagService:
    def __init__(self):
        self.agent_with_history = None
        self.vectorstore = get_vector_store()

    def _limpar_texto(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'^\|[\s\d\|-]+\|$', '', text, flags=re.MULTILINE)
        text = re.sub(r"UNIVERSIDADE ESTADUAL DO MARANHÃO|www\.uema\.br", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def ingerir_base_conhecimento(self):
        """Lê arquivos da pasta 'dados/' e popula o banco vetorial."""
        data_dir = getattr(settings, "DATA_DIR", "/app/dados")
        
        try:
            if len(self.vectorstore.similarity_search("UEMA", k=1)) > 0:
                print("💾 Banco Vetorial já populado. Pulando ingestão.")
                return
        except Exception:
            pass

        print(f"🕵️ Iniciando ingestão da pasta: {data_dir}")
        arquivos = glob.glob(os.path.join(data_dir, "*.[pP][dD][fF]")) + \
                   glob.glob(os.path.join(data_dir, "*.[tT][xX][tT]"))

        if not arquivos:
            print("⚠️ Nenhum arquivo encontrado em 'dados/'.")
            return

        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            language="pt",
            verbose=False
        )

        all_documents = []
        for arquivo in arquivos:
            print(f"📦 Processando: {os.path.basename(arquivo)}...")
            try:
                docs = parser.load_data(arquivo)
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(arquivo)
                    doc.text = self._limpar_texto(doc.text)
                all_documents.extend(docs)
            except Exception as e:
                print(f"❌ Erro ao ler {arquivo}: {e}")

        if all_documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            final_chunks = splitter.split_documents(all_documents)
            self.vectorstore.add_documents(final_chunks)
            print(f"✅ {len(final_chunks)} chunks salvos no banco!")

    def inicializar(self):
        """Configura o agente, as tools importadas e a blindagem de erro."""
        print("🧠 Inicializando Agente de IA...")
        self.ingerir_base_conhecimento()

        # 1. Carrega a sua Tool de Calendário Modular (mmr, k=5, etc.)
        tool_calendario = get_calendar_tool()
        tool_calendario.handle_tool_error = handle_tool_error

        # 2. Tool de Conhecimento Geral (RU, Contatos, Locais)
        # Usamos uma busca mais enxuta para evitar Rate Limits
        retriever_geral = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6}
        )
        tool_geral = create_retriever_tool(
            retriever_geral,
            "consultar_base_geral",
            "Busque aqui sobre RU, Ônibus, Contatos e Emails. Não use para Calendário."
        )
        tool_geral.handle_tool_error = handle_tool_error

        # 3. Tool GLPI (Blindada)
        tool_glpi = StructuredTool.from_function(
            func=abrir_chamado_glpi,
            name="abrir_chamado_glpi",
            description="Abre chamados de suporte técnico no GLPI.",
            handle_tool_error=handle_tool_error
        )

        # Agrupamento das Ferramentas
        tools = [tool_calendario, tool_geral, tool_glpi, consultar_fila, enviar_email_notificacao]

        # LLM Principal
        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )

        # O System Prompt "Foda" (O Oráculo Ludovicense)
        system_prompt = """
        Você é o Assistente Virtual Oficial da UEMA (Campus Paulo VI - São Luís).
        Atue como um facilitador de vida acadêmica.

        📍 GEOLOCALIZAÇÃO: Assuma sempre São Luís por padrão.
        
        🧠 LOGICA DE FERRAMENTAS:
        - Para datas, prazos e feriados, use 'consultar_calendario_academico'.
        - Para RU, Ônibus e Telefones, use 'consultar_base_geral'.
        - Se uma ferramenta retornar 'ERRO TÉCNICO', peça desculpas e não tente a mesma busca.
        - Se a busca não trouxer dados, admita que a informação não consta nos manuais oficiais.

        🛡️ POSTURA: Profissional, acolhedor e focado em resolver.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Criação do Agente com controle de iterações para evitar loops
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=5
        )

        # Integração com Memória Redis
        self.agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        print("✅ Agente e Ferramentas Prontos!")

    def responder(self, texto: str, user_id: str):
        if not self.agent_with_history:
            return "⚠️ Sistema em aquecimento. Tente novamente em 10 segundos."
            
        config = {"configurable": {"session_id": user_id}}
        try:
            resultado = self.agent_with_history.invoke({"input": texto}, config=config)
            return resultado["output"]
        except Exception as e:
            logger.log_error(user_id, "Critical Response Failure", str(e))
            return "Desculpe, tive uma dificuldade técnica agora. Podemos tentar novamente?"