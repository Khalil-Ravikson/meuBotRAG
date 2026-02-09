import os
import nest_asyncio
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

# Importação correta da Community
from langchain_community.chat_message_histories import RedisChatMessageHistory

from langchain_core.tools import create_retriever_tool
from langchain_core.documents import Document

# Ingestão Avançada
from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Outros
from src.config import settings
from src.services.db_service import get_vector_store
from src.tools import abrir_chamado_glpi, consultar_fila
from groq import Groq 

# Aplica nest_asyncio
nest_asyncio.apply()

class RagService:
    def __init__(self):
        self.agent_with_history = None
        self.vectorstore = get_vector_store()

    # --- 1. INGESTÃO INTELIGENTE (LlamaParse) ---
    def ingerir_pdf(self):
        try:
            # Busca dummy para checar se o banco está vazio
            if len(self.vectorstore.similarity_search("calendário", k=1)) > 0:
                print("💾 Banco de dados já populado. Pulando ingestão.")
                return
        except Exception as e:
            print(f"⚠️ Falha na checagem do Banco (normal na primeira execução): {e}")
            pass

        if not os.path.exists(settings.PDF_PATH):
            print(f"⚠️ PDF não encontrado: {settings.PDF_PATH}")
            return

        print("🕵️ LlamaParse: Convertendo PDF para Markdown estruturado...")
        try:
            # CONFIGURAÇÃO CORRETA DO PARSER (Aqui que vai o result_type)
            parser = LlamaParse(
                api_key=settings.LLAMA_CLOUD_API_KEY,
                result_type="markdown",  # <--- CRÍTICO PARA LER TABELAS
                verbose=True,
                language="pt"
            )
            llama_docs = parser.load_data(settings.PDF_PATH)
            
            if not llama_docs:
                print("❌ LlamaParse não retornou conteúdo.")
                return

            texto_completo = llama_docs[0].text
            
            # Corta por Cabeçalhos (Melhor para tabelas e docs estruturados)
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2")])
            chunks = splitter.split_text(texto_completo)
            
            # Adiciona Metadados
            for c in chunks: 
                c.metadata["source"] = "calendario_2026"

            self.vectorstore.add_documents(chunks)
            print(f"✅ {len(chunks)} blocos estruturados salvos no Postgres!")
            
        except Exception as e:
            print(f"❌ Erro durante a ingestão: {e}")

    # --- 2. TRANSCRIÇÃO DE ÁUDIO ---
    def transcrever_audio(self, caminho_arquivo):
        print(f"🎧 Transcrevendo: {caminho_arquivo}")
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)
            with open(caminho_arquivo, "rb") as file:
                return client.audio.transcriptions.create(
                    file=(caminho_arquivo, file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
        except Exception as e:
            print(f"❌ Erro ao transcrever áudio: {e}")
            return "Erro ao processar áudio."

    # --- 3. INICIALIZAÇÃO DO AGENTE ---
    def get_session_history(self, session_id: str):
        return RedisChatMessageHistory(session_id, url=settings.REDIS_URL, ttl=3600)

    def inicializar(self):
        print("🧠 Inicializando Agente de IA...")

        # Transforma o Banco Vetorial na Ferramenta 'buscar_no_calendario'
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        tool_pdf = create_retriever_tool(
            retriever,
            "buscar_no_calendario",
            "Use para buscar datas, feriados, prazos e regras no calendário acadêmico oficial."
        )

        # Lista correta de ferramentas
        tools = [tool_pdf, abrir_chamado_glpi, consultar_fila]

        # LLM conectada às ferramentas
        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.3 # Mais baixo = Mais sério/preciso
        ).bind_tools(tools)

        # Prompt do Sistema (Blindado e Sério)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é o Assistente Virtual Institucional da UEMA (Universidade Estadual do Maranhão).
            Sua postura é ESTRITAMENTE profissional, objetiva e impessoal.
            
            🚨 REGRAS DE OURO (ESCOPO):
            1. O seu ÚNICO objetivo é auxiliar com: Calendário Acadêmico, Suporte Técnico (GLPI) e Processos da UEMA.
            2. Se o usuário falar sobre QUALQUER assunto externo (política, futebol, promoções, receitas, piadas, clima, fofoca), você DEVE responder:
               "Desculpe, meu escopo de atuação limita-se exclusivamente a assuntos acadêmicos e técnicos da UEMA."
            3. NÃO emita opiniões pessoais e NÃO tente ser engraçado.
            
            🛠️ INSTRUÇÕES DE FERRAMENTAS:
            - Perguntas sobre Datas, Prazos ou Feriados: Você É OBRIGADO a usar a ferramenta 'buscar_no_calendario'. Não invente datas.
            - Relato de Problemas (PC quebrou, sem internet): Use 'abrir_chamado_glpi'.
            - Consultas de Status: Use 'consultar_fila'.
            
            👋 SAUDAÇÕES:
            - Se o usuário disser "Oi", "Bom dia", etc: Responda apenas: "Olá. Sou o assistente da UEMA. Em que posso ajudar referente à universidade?"
            
            Seja breve. Não enrole."""),
            MessagesPlaceholder(variable_name="history"), 
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"), 
            ])

        # Cria o Agente
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Adiciona Memória (Redis)
        self.agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        print("✅ Agente Pronto!")

    def responder(self, texto: str, user_id: str):
        if self.agent_with_history is None:
            return "⚠️ O sistema está iniciando, por favor tente novamente em alguns segundos."
            
        config = {"configurable": {"session_id": user_id}}
        
        try:
            resultado = self.agent_with_history.invoke(
                {"input": texto},
                config=config
            )
            return resultado["output"]
        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            return "Desculpe, tive um erro interno ao processar seu pedido."