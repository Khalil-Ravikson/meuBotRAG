"""
================================================================================
rag_service.py — v5: Correções críticas de histórico e output do agente
================================================================================

CORREÇÕES NESTA VERSÃO:

  1. ERRO CRÍTICO CORRIGIDO: "Direct assignment to 'messages' is not allowed"
     ─────────────────────────────────────────────────────────────────────────
     O RedisChatMessageHistory NÃO permite history.messages = [...].
     Solução: reconstruir o objeto deletando mensagens antigas via Redis diretamente,
     ou usar uma subclasse que sobrescreve o setter.
     Implementamos a solução correta: apaga as mensagens antigas pelo Redis
     e re-adiciona apenas as N mais recentes.

  2. ERRO CRÍTICO CORRIGIDO: "Agent stopped due to max iterations." enviado ao usuário
     ─────────────────────────────────────────────────────────────────────────
     Quando o agente atinge max_iterations, o LangChain retorna essa string
     literal como output. Precisamos interceptar e converter em mensagem amigável.
     Solução: _sanitizar_output() detecta e substitui outputs inválidos.

  3. FERRAMENTA RETORNANDO "Não encontrei" — DIAGNÓSTICO
     ─────────────────────────────────────────────────────────────────────────
     Causa provável: o filtro {"source": "nome_exato.pdf"} não bate com o
     metadado real dos chunks no banco. O nome do arquivo durante a ingestão
     pode ser diferente do esperado.
     Solução: diagnose_banco() imprime os sources reais presentes no banco.
     Use isso para confirmar os nomes e ajustar PDF_CONFIG / SOURCE_* nas tools.

  4. Rate limit 429 do Groq
     ─────────────────────────────────────────────────────────────────────────
     O agente fazia 4+ chamadas por mensagem (cada tool call = 1 chamada).
     O plano free do Groq tem limite de 12.000 tokens/min.
     Solução: max_iterations reduzido para 3, e adicionado tratamento explícito
     do erro 429 com mensagem amigável ao usuário.
================================================================================
"""

import os
import re
import glob
import logging

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse

from src.tools.calendar_tool import get_tool_calendario
from src.tools.tool_edital import get_tool_edital
from src.tools.tool_contatos import get_tool_contatos

from src.services.logger_service import LogService
from src.config import settings
from src.services.db_service import get_vector_store
from src.services.redis_history import get_session_history, limpar_historico

logger = logging.getLogger(__name__)
log_service = LogService()

MAX_HISTORY_MESSAGES = 6  # reduzido: menos tokens, menos rate limit

PDF_CONFIG = {
    "calendario-academico-2026.pdf": {
        "chunk_size": 400,
        "chunk_overlap": 50,
        "parsing_instruction": (
            "Este PDF é o Calendário Acadêmico da UEMA 2026. "
            "Para CADA linha de evento na tabela, formate assim:\n"
            "EVENTO: [nome do evento] | DATA: [data ou período] | SEM: [semestre]\n"
            "Exemplo: EVENTO: Matrícula de veteranos | DATA: 03/02/2026 a 07/02/2026 | SEM: 2026.1\n"
            "Mantenha TODOS os eventos e datas."
        ),
    },
    "edital_paes_2026.pdf": {
        "chunk_size": 600,
        "chunk_overlap": 80,
        "parsing_instruction": (
            "Este PDF é o Edital do Processo Seletivo PAES 2026 da UEMA. "
            "Para tabelas de vagas, preserve:\n"
            "CURSO: [nome] | TURNO: [turno] | AC: [nº] | PcD: [nº] | TOTAL: [nº]\n"
            "Para categorias de cotas:\n"
            "CATEGORIA: [sigla] | NOME: [nome completo] | PÚBLICO: [descrição]\n"
            "Preserve todos os números de vagas e numeração dos itens."
        ),
    },
    "guia_contatos_2025.pdf": {
        "chunk_size": 300,
        "chunk_overlap": 30,
        "parsing_instruction": (
            "Este PDF é o Guia de Contatos da UEMA 2025. "
            "Para cada linha de contato, formate:\n"
            "CARGO: [cargo] | NOME: [nome completo] | EMAIL: [email] | TEL: [telefone]\n"
            "Exemplo: CARGO: Diretor CECEN | NOME: Regina Célia | EMAIL: cecen@uema.br | TEL: (98) 99232-4837\n"
            "Mantenha o nome do centro/unidade como cabeçalho de cada bloco."
        ),
    },
}


def handle_tool_error(error: Exception) -> str:
    log_service.log_error("SYSTEM", "Tool Error", str(error))
    return (
        "ERRO TÉCNICO NA FERRAMENTA. "
        "Não tente esta ferramenta novamente. "
        "Informe ao usuário que o sistema está temporariamente instável."
    )


# =============================================================================
# Truncamento correto do histórico Redis
# =============================================================================

def get_session_history_limitado(session_id: str):
    """
    Retorna o histórico da sessão com no máximo MAX_HISTORY_MESSAGES mensagens.

    CORREÇÃO: RedisChatMessageHistory não permite atribuição direta a .messages.
    Solução correta: apagar as mensagens excedentes pelo Redis e re-adicionar
    apenas as recentes, usando os métodos públicos da classe.
    """
    history = get_session_history(session_id)

    try:
        msgs = history.messages  # lê a lista atual
        if len(msgs) > MAX_HISTORY_MESSAGES:
            excesso = len(msgs) - MAX_HISTORY_MESSAGES
            msgs_recentes = msgs[excesso:]  # mantém as N mais recentes

            # Limpa o histórico e re-adiciona apenas as mensagens recentes
            # history.clear() apaga tudo; add_messages() re-adiciona
            history.clear()
            history.add_messages(msgs_recentes)

            logger.debug(
                "✂️  Histórico [%s] truncado: %d removidas, %d mantidas.",
                session_id, excesso, len(msgs_recentes)
            )
    except Exception as e:
        # Se falhar o truncamento, não quebra — só loga e segue com histórico cheio
        logger.warning("⚠️  Falha ao truncar histórico [%s]: %s", session_id, e)

    return history


# =============================================================================
# Sanitização do output do agente
# =============================================================================

# Strings que o LangChain retorna quando o agente falha internamente
_OUTPUTS_INVALIDOS = {
    "agent stopped due to max iterations.",
    "agent stopped due to iteration limit or time limit.",
    "parsing error",
}

def _sanitizar_output(output: str) -> str:
    """
    Intercepta outputs internos do LangChain que não devem ser enviados ao usuário.

    Quando o agente atinge max_iterations sem concluir, o LangChain retorna
    literalmente "Agent stopped due to max iterations." como output.
    Isso jamais deve ser enviado ao WhatsApp.
    """
    if not output:
        return ""

    output_lower = output.strip().lower()
    for invalido in _OUTPUTS_INVALIDOS:
        if invalido in output_lower:
            logger.warning("⚠️  Output inválido do agente interceptado: '%s'", output[:80])
            return (
                "Não consegui encontrar essa informação no momento. "
                "Tente reformular sua pergunta ou use o menu para escolher uma área."
            )
    return output


# =============================================================================
# RagService
# =============================================================================

class RagService:
    def __init__(self):
        self.agent_with_history = None
        self.agent_executor = None
        self.vectorstore = get_vector_store()

    # =========================================================================
    # Diagnóstico do banco vetorial
    # =========================================================================

    def diagnose_banco(self):
        """
        Imprime os 'source' únicos presentes no banco vetorial.

        USE ISSO quando as tools retornam "Não encontrei":
        Os nomes aqui devem bater EXATAMENTE com SOURCE_* em cada tool
        e com as chaves de PDF_CONFIG.

        Exemplo de uso: chame no startup após inicializar().
        """
        try:
            # Busca genérica para trazer documentos de qualquer fonte
            docs = self.vectorstore.similarity_search("UEMA", k=50)
            sources = set(doc.metadata.get("source", "SEM_SOURCE") for doc in docs)
            print("=" * 60)
            print("🔍 DIAGNÓSTICO DO BANCO VETORIAL")
            print(f"   Chunks encontrados: {len(docs)}")
            print(f"   Sources presentes: {sources}")
            print("   ⚠️  Os nomes acima devem bater com:")
            print(f"      PDF_CONFIG keys: {list(PDF_CONFIG.keys())}")
            print("=" * 60)
            return sources
        except Exception as e:
            print(f"❌ Diagnóstico falhou: {e}")
            return set()

    # =========================================================================
    # Limpeza de texto
    # =========================================================================

    def _limpar_texto(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"^\|[\s\d\|\-:]+\|$", "", text, flags=re.MULTILINE)
        text = re.sub(
            r"UNIVERSIDADE ESTADUAL DO MARANHÃO|www\.uema\.br|UEMA\s*[-–]\s*Campus",
            "", text, flags=re.IGNORECASE,
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # =========================================================================
    # Verificação do banco
    # =========================================================================

    def _banco_ja_populado(self) -> bool:
        try:
            if self.vectorstore.similarity_search("UEMA 2026", k=1):
                return True
        except Exception as e:
            logger.warning("⚠️  similarity_search falhou: %s", e)
        try:
            if self.vectorstore._collection.count() > 0:
                return True
        except Exception:
            pass
        return False

    # =========================================================================
    # Ingestão
    # =========================================================================

    def ingerir_base_conhecimento(self):
        """
        Processa os PDFs com parsing_instruction específica por arquivo.
        O metadado 'source' salvo deve bater EXATAMENTE com SOURCE_* nas tools.
        """
        data_dir = getattr(settings, "DATA_DIR", "/app/dados")

        if self._banco_ja_populado():
            print("💾 Banco Vetorial já populado. Pulando ingestão.")
            return

        print(f"🕵️  Iniciando ingestão em: {data_dir}")
        arquivos_pdf = glob.glob(os.path.join(data_dir, "*.[pP][dD][fF]"))

        if not arquivos_pdf:
            print("⚠️  Nenhum PDF encontrado.")
            return

        print(f"📁 PDFs: {[os.path.basename(a) for a in arquivos_pdf]}")

        for arquivo in arquivos_pdf:
            nome = os.path.basename(arquivo)
            config = PDF_CONFIG.get(nome)

            if not config:
                print(f"⚠️  '{nome}' não está no PDF_CONFIG. Pulando.")
                print(f"   Esperados: {list(PDF_CONFIG.keys())}")
                continue

            print(f"📦 Processando '{nome}'...")

            parser = LlamaParse(
                api_key=settings.LLAMA_CLOUD_API_KEY,
                result_type="markdown",
                language="pt",
                verbose=False,
                parsing_instruction=config["parsing_instruction"],
            )

            try:
                llama_docs = parser.load_data(arquivo)
                documentos: list[Document] = []

                for llama_doc in llama_docs:
                    texto = self._limpar_texto(llama_doc.text)
                    if not texto:
                        continue
                    documentos.append(Document(
                        page_content=texto,
                        metadata={
                            "source": nome,  # ← deve bater com SOURCE_* nas tools
                            **{k: v for k, v in (llama_doc.metadata or {}).items()
                               if isinstance(v, (str, int, float, bool))},
                        },
                    ))

                if not documentos:
                    print(f"⚠️  Nenhum conteúdo extraído de '{nome}'.")
                    continue

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config["chunk_size"],
                    chunk_overlap=config["chunk_overlap"],
                    separators=["\n\n", "\n", " ", ""],
                )
                chunks = splitter.split_documents(documentos)
                self.vectorstore.add_documents(chunks)
                print(f"✅ '{nome}': {len(chunks)} chunks salvos.")

            except Exception as e:
                print(f"❌ Erro em '{nome}': {e}")
                logger.exception("Ingestão falhou para '%s'", nome)

        print("✅ Ingestão concluída.")
        # Diagnóstico automático após ingestão para confirmar sources
        self.diagnose_banco()

    # =========================================================================
    # Inicialização do agente
    # =========================================================================

    def inicializar(self):
        print("🧠 Inicializando Agente...")
        self.ingerir_base_conhecimento()

        tool_calendario = get_tool_calendario()
        tool_calendario.handle_tool_error = handle_tool_error

        tool_edital = get_tool_edital()
        tool_edital.handle_tool_error = handle_tool_error

        tool_contatos = get_tool_contatos()
        tool_contatos.handle_tool_error = handle_tool_error

        tools = [tool_calendario, tool_edital, tool_contatos]

        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
        )

        system_prompt = """Você é o Assistente Virtual da UEMA (Universidade Estadual do Maranhão), \
Campus Paulo VI, São Luís - MA.
Responda sempre em português brasileiro, de forma objetiva e precisa.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FERRAMENTAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 consultar_calendario_academico
   Para: datas do calendário letivo 2026 (matrícula, prova, feriado, semestre, trancamento)
   Query: "matricula veteranos 2026.1" | "feriados marco" | "inicio aulas"

📋 consultar_edital_paes_2026
   Para: processo seletivo PAES 2026 (vagas, cotas, inscrição, documentos, cronograma)
   Query: "vagas engenharia civil" | "documentos inscricao" | "cotas BR-PPI"

📞 consultar_contatos_uema
   Para: e-mails, telefones, responsáveis de setores da UEMA
   Query: "PROG pro-reitoria email" | "CTIC TI contato" | "CECEN diretor"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Use APENAS o retorno das ferramentas. NUNCA invente datas, vagas ou contatos.
2. Se a ferramenta retornar "Não encontrei": tente UMA query diferente. Se ainda não encontrar, diga ao usuário que a informação não está disponível no momento e sugira uema.br.
3. Se retornar "ERRO TÉCNICO": responda "Tive uma instabilidade. Tente em instantes." e PARE.
4. Máximo de 2 tentativas por ferramenta. Depois, responda com o que encontrou ou informe que não encontrou.
5. Respostas curtas: até 3 parágrafos ou 6 itens em lista.
6. Use *negrito* para datas, e-mails e setores.
7. Se uma ferramenta retornar nada ou falhar retorne ao menu inicial e iniciando uma nova conversa sem dados da anterior


"""

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
            max_iterations=3,        # reduzido: 1 tool call + raciocínio = 2 steps
            max_execution_time=25,   # timeout de 25s por resposta
            return_intermediate_steps=False,
        )

        self.agent_with_history = RunnableWithMessageHistory(
            self.agent_executor,
            get_session_history_limitado,
            input_messages_key="input",
            history_messages_key="history",
        )

        print("✅ Agente pronto!")

    # =========================================================================
    # Resposta
    # =========================================================================

    def responder(self, texto: str, user_id: str) -> str:
        if not self.agent_with_history:
            return "⚠️ Sistema em aquecimento. Tente novamente em 10 segundos."

        config = {"configurable": {"session_id": user_id}}

        try:
            resultado = self.agent_with_history.invoke({"input": texto}, config=config)
            output = resultado.get("output", "")

            # Intercepta outputs inválidos do LangChain antes de enviar ao usuário
            return _sanitizar_output(output)

        except Exception as e:
            erro_str = str(e)

            # Rate limit 429 do Groq
            if "429" in erro_str or "rate_limit" in erro_str.lower() or "Too Many Requests" in erro_str:
                log_service.log_warn(user_id, "Rate limit Groq", erro_str[:200])
                return (
                    "O sistema está com alta demanda no momento. "
                    "Aguarde alguns segundos e tente novamente. 🙏"
                )

            # tool_use_failed → limpa histórico corrompido e tenta sem ele
            if "400" in erro_str and "tool_use_failed" in erro_str:
                log_service.log_warn(user_id, "tool_use_failed — limpando histórico", erro_str[:200])
                limpar_historico(user_id)
                try:
                    resultado = self.agent_executor.invoke({"input": texto, "history": []})
                    return _sanitizar_output(resultado.get("output", ""))
                except Exception as e2:
                    log_service.log_error(user_id, "Fallback falhou", str(e2)[:200])
                    return "Desculpe, tive uma instabilidade. Seu histórico foi reiniciado. Pode repetir?"

            log_service.log_error(user_id, "Erro crítico na resposta", erro_str[:300])
            return "Desculpe, tive uma dificuldade técnica. Tente novamente."