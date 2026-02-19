"""
rag_service.py — v2 com interceptação de tool_use_failed

Problema raiz do erro 400:
  O LLaMA 3.3 no Groq gera tool calls com JSON mal-formatado quando o
  contexto está muito cheio (ex: chunks grandes do RAG). O LangChain salva
  a AIMessage corrompida no Redis. Na próxima mensagem, o Groq recebe esse
  histórico inválido e falha de novo — criando o loop de erros.

Solução aplicada aqui:
  1. calendar_tool agora trunca respostas (MAX_CHARS=900) → contexto menor
  2. responder() detecta erro 400 do Groq, limpa o histórico corrompido
     automaticamente e tenta de novo SEM histórico (modo fallback)
  3. Se o fallback também falhar, retorna mensagem amigável
"""

import os
import re
import glob

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import create_retriever_tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse

from src.tools.calendar_tool import get_calendar_tool
from src.tools.email_tool import enviar_email_notificacao
from src.tools.fila import consultar_fila
from src.tools.glpi import abrir_chamado_glpi
from src.services.logger_service import LogService
from src.config import settings
from src.services.db_service import get_vector_store
from src.services.redis_history import get_session_history, limpar_historico

logger = LogService()


def handle_tool_error(error: Exception) -> str:
    logger.log_error("SYSTEM", "Tool Execution Failure", str(error))
    return (
        "ERRO TÉCNICO NA FERRAMENTA. "
        "Não tente esta ferramenta novamente. "
        "Informe ao usuário que o sistema está temporariamente instável."
    )


class RagService:
    def __init__(self):
        self.agent_with_history = None
        self.agent_executor = None          # guardamos separado para o fallback
        self.vectorstore = get_vector_store()

    # ------------------------------------------------------------------
    # Limpeza de texto
    # ------------------------------------------------------------------

    def _limpar_texto(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'^\|[\s\d\|-]+\|$', '', text, flags=re.MULTILINE)
        text = re.sub(
            r"UNIVERSIDADE ESTADUAL DO MARANHÃO|www\.uema\.br",
            "", text, flags=re.IGNORECASE,
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Verificação segura do banco
    # ------------------------------------------------------------------

    def _banco_ja_populado(self) -> bool:
        try:
            if self.vectorstore.similarity_search("UEMA calendário", k=1):
                return True
        except Exception as e:
            print(f"⚠️  similarity_search falhou: {e}")
        try:
            if self.vectorstore._collection.count() > 0:
                return True
        except AttributeError:
            pass
        except Exception as e:
            print(f"⚠️  collection.count falhou: {e}")
        return False

    # ------------------------------------------------------------------
    # Ingestão
    # ------------------------------------------------------------------

    def ingerir_base_conhecimento(self):
        data_dir = getattr(settings, "DATA_DIR", "/app/dados")

        if self._banco_ja_populado():
            print("💾 Banco Vetorial já populado. Pulando ingestão.")
            return

        print(f"🕵️  Iniciando ingestão: {data_dir}")
        arquivos = (
            glob.glob(os.path.join(data_dir, "*.[pP][dD][fF]"))
            + glob.glob(os.path.join(data_dir, "*.[tT][xX][tT]"))
        )

        if not arquivos:
            print("⚠️  Nenhum arquivo encontrado.")
            return

        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            language="pt",
            verbose=False,
        )

        all_documents: list[Document] = []
        for arquivo in arquivos:
            nome = os.path.basename(arquivo)
            print(f"📦 Processando: {nome}...")
            try:
                llama_docs = parser.load_data(arquivo)
                for llama_doc in llama_docs:
                    texto = self._limpar_texto(llama_doc.text)
                    if not texto:
                        continue
                    all_documents.append(Document(
                        page_content=texto,
                        metadata={
                            "source": nome,
                            **{k: v for k, v in (llama_doc.metadata or {}).items()
                               if isinstance(v, (str, int, float, bool))},
                        },
                    ))
            except Exception as e:
                print(f"❌ Erro em '{nome}': {e}")

        if not all_documents:
            print("⚠️  Nenhum conteúdo extraído.")
            return

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=600,     # reduzido: chunks menores → menos contexto por chamada
            chunk_overlap=80,
        ).split_documents(all_documents)

        self.vectorstore.add_documents(chunks)
        print(f"✅ {len(chunks)} chunks salvos!")

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    def inicializar(self):
        print("🧠 Inicializando Agente...")
        self.ingerir_base_conhecimento()

        tool_calendario = get_calendar_tool()
        tool_calendario.handle_tool_error = handle_tool_error

        retriever_geral = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.6},
        )
        tool_geral = create_retriever_tool(
            retriever_geral,
            "consultar_base_geral",
            (
                "Consulta sobre RU, ônibus, contatos e e-mails da UEMA. "
                "NÃO use para calendário ou datas acadêmicas."
            ),
        )
        tool_geral.handle_tool_error = handle_tool_error

        tool_glpi = abrir_chamado_glpi
        tool_glpi.handle_tool_error = handle_tool_error

        tools = [tool_calendario, tool_geral, tool_glpi, consultar_fila, enviar_email_notificacao]

        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )

        # System prompt enxuto — menos tokens = mais espaço para tool calls
        system_prompt = """Você é o Assistente Virtual da UEMA (Campus Paulo VI, São Luís).
Seja objetivo e preciso.

ROTEAMENTO DE FERRAMENTAS:
- Datas, prazos, matrículas, feriados → consultar_calendario_academico
- RU, ônibus, telefones, e-mails → consultar_base_geral
- Chamados de TI → abrir_chamado_glpi
- Fila de atendimento → consultar_fila
- Envio de e-mail → enviar_email_notificacao

REGRAS:
- Use APENAS o retorno das ferramentas, nunca invente datas.
- Se uma ferramenta retornar ERRO TÉCNICO, informe o usuário e não tente novamente.
- Respostas curtas e diretas."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=4,           # reduzido: evita loops longos
        )

        self.agent_with_history = RunnableWithMessageHistory(
            self.agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        print("✅ Agente pronto!")

    # ------------------------------------------------------------------
    # Resposta com auto-recuperação de tool_use_failed
    # ------------------------------------------------------------------

    def responder(self, texto: str, user_id: str) -> str:
        if not self.agent_with_history:
            return "⚠️ Sistema em aquecimento. Tente em 10 segundos."

        config = {"configurable": {"session_id": user_id}}

        try:
            resultado = self.agent_with_history.invoke({"input": texto}, config=config)
            return resultado["output"]

        except Exception as e:
            erro_str = str(e)

            # ── Detecta o erro 400 de tool_use_failed do Groq ──────────
            if "400" in erro_str and "tool_use_failed" in erro_str:
                logger.log_error(user_id, "tool_use_failed detectado — limpando histórico", erro_str)

                # Limpa o histórico corrompido
                limpar_historico(user_id)

                # Tenta responder de novo SEM histórico (contexto zerado)
                try:
                    resultado = self.agent_executor.invoke({"input": texto, "history": []})
                    return resultado["output"]
                except Exception as e2:
                    logger.log_error(user_id, "Fallback também falhou", str(e2))
                    return (
                        "Desculpe, tive uma instabilidade técnica. "
                        "Seu histórico foi reiniciado automaticamente — pode repetir a pergunta?"
                    )

            # ── Outros erros ────────────────────────────────────────────
            logger.log_error(user_id, "Critical Response Failure", erro_str)
            return "Desculpe, tive uma dificuldade técnica. Podemos tentar novamente?"