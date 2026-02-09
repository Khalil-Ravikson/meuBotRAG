import os
import nest_asyncio
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory





# Importação correta da Community
from langchain_community.chat_message_histories import RedisChatMessageHistory

from langchain_core.tools import create_retriever_tool

# Ingestão Avançada
from llama_parse import LlamaParse

# --- imports novos no topo do arquivo ---
import re
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_core.documents import Document

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
            # Checa se o banco já tem algo
            if len(self.vectorstore.similarity_search("calendário", k=1)) > 0:
                print("💾 Banco já populado. Pulando ingestão.")
                return
        except Exception:
            pass

        if not os.path.exists(settings.PDF_PATH):
            print(f"⚠️ PDF não encontrado: {settings.PDF_PATH}")
            return

        print("🕵️ LlamaParse: Convertendo PDF com parsing inteligente...")

        # 🔥 SYSTEM PROMPT (igual ao do Colab)
        system_prompt = """
        Este é um calendário acadêmico.
        IMPORTANTE:
        1. IGNORE grades visuais mensais que contenham apenas números de dias (1, 2, 3...).
        2. Extraia APENAS texto relacionado a eventos, feriados, prazos, início e fim de períodos.
        3. Converta tabelas relevantes (atividades, público-alvo) em Markdown limpo.
        4. Ignore cabeçalhos e rodapés institucionais repetitivos.
        """

        try:
            parser = LlamaParse(
                api_key=settings.LLAMA_CLOUD_API_KEY,
                result_type="markdown",
                language="pt",
                system_prompt=system_prompt,
                verbose=True
            )

            llama_docs = parser.load_data(settings.PDF_PATH)

            if not llama_docs:
                print("❌ LlamaParse não retornou conteúdo.")
                return

            # --- FUNÇÃO DE LIMPEZA ---
            def clean_text(text: str) -> str:
                if not text:
                    return ""

                # Remove pseudo-tabelas só com números
                text = re.sub(r'^\|[\s\d\|-]+\|$', '', text, flags=re.MULTILINE)

                # Remove lixo institucional
                patterns = [
                    r"UNIVERSIDADE ESTADUAL DO MARANHÃO",
                    r"Pró-Reitoria de Graduação",
                    r"Cidade Universitária Paulo VI",
                    r"www\.uema\.br",
                ]
                for p in patterns:
                    text = re.sub(p, "", text, flags=re.IGNORECASE)

                # Normaliza quebras de linha
                text = re.sub(r"\n{3,}", "\n\n", text)
                return text.strip()

            # --- SPLITTER POR CABEÇALHOS ---
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "contexto_macro"),
                    ("##", "secao_referencia"),
                    ("###", "topico_especifico"),
                ]
            )

            # --- SPLITTER POR TAMANHO ---
            size_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", "###", "##"]
            )

            all_chunks: list[Document] = []

            print("📦 Processando e limpando chunks...")

            for doc in llama_docs:
                cleaned_text = clean_text(doc.text)

                if not cleaned_text:
                    continue

                # Split semântico
                header_docs = header_splitter.split_text(cleaned_text)

                # Injeta metadados
                for hdoc in header_docs:
                    hdoc.metadata.update(doc.metadata)
                    hdoc.metadata["source"] = "calendario_2026"

                # Split final por tamanho
                final_chunks = size_splitter.split_documents(header_docs)
                all_chunks.extend(final_chunks)

            if not all_chunks:
                print("⚠️ Nenhum chunk útil gerado.")
                return

            self.vectorstore.add_documents(all_chunks)
            print(f"✅ {len(all_chunks)} chunks limpos salvos no Postgres!")

        except Exception as e:
            print(f"❌ Erro durante ingestão avançada: {e}")


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

        # --- Retriever ---
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={        
                            "k": 5,
                            "fetch_k": 20,
                            "lambda_mult": 0.5
                            })
        tool_pdf = create_retriever_tool(
            retriever,
            "buscar_no_calendario",
            "Use para buscar datas, feriados, prazos e regras no calendário acadêmico oficial."
        )

        tools = [tool_pdf, abrir_chamado_glpi, consultar_fila]

        # --- LLM ---
        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )

        # --- Agente ---
        agent_executor = self._criar_agente(llm, tools)

        # --- Memória Redis ---
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
        
        
    
    
    def _criar_agente(self, llm, tools):
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
                - Se o usuário disser "Oi", "Bom dia", etc: Responda apenas:
                "Olá. Sou o assistente da UEMA. Em que posso ajudar referente à universidade?"

                Seja breve. Não enrole."""),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])

        agent = create_tool_calling_agent(llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )

    