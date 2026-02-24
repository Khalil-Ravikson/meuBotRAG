"""
agent/core.py — Orquestrador do agente multi-step
==================================================
Substitui a parte de "agente" do rag_service.py.

Responsabilidades:
  - Montar o AgentExecutor LangChain (com tools + LLM + prompt)
  - Executar o loop: invoke → validate → (retry | done)
  - Tratar rate limit 429 e tool_use_failed
  - Retornar AgentResponse com métricas

NÃO faz ingestão (→ rag/ingestor.py)
NÃO faz busca vetorial (→ rag/retriever.py)
NÃO tem lógica de menu (→ domain/menu.py)
"""
from __future__ import annotations
import logging
import time

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.agent.state import AgentState
from src.agent.validator import validar
from src.agent.prompts import (
    SYSTEM_PROMPT, MSG_RATE_LIMIT, MSG_ERRO_TECNICO, MSG_HISTORICO_RESETADO
)
from src.domain.entities import AgentResponse, Rota
from src.providers.groq_provider import get_llm
from src.memory.redis_memory import get_historico_limitado, limpar_historico
from src.infrastructure.settings import settings
from src.infrastructure.observability import obs

logger = logging.getLogger(__name__)


class AgentCore:
    """
    Singleton do agente LangChain.
    Inicializado uma vez no startup, reutilizado para todas as mensagens.
    """

    def __init__(self):
        self._agent_with_history = None
        self._agent_executor     = None
        self._tools              = []

    def inicializar(self, tools: list) -> None:
        """
        Monta o AgentExecutor. Chamado no startup após a ingestão dos PDFs.

        Parâmetro:
          tools: lista de LangChain tools (vem de src/tools/__init__.py)
        """
        self._tools = tools

        llm = get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)

        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=settings.AGENT_MAX_ITERATIONS,
            max_execution_time=settings.AGENT_TIMEOUT_S,
            return_intermediate_steps=False,
        )

        self._agent_with_history = RunnableWithMessageHistory(
            self._agent_executor,
            get_historico_limitado,
            input_messages_key="input",
            history_messages_key="history",
        )

        logger.info("✅ AgentCore inicializado com %d tools.", len(tools))

    def responder(self, state: AgentState) -> AgentResponse:
        """
        Executa o agente para o estado dado e retorna AgentResponse.
        Trata rate limit e tool_use_failed internamente.
        """
        if not self._agent_with_history:
            return AgentResponse(
                conteudo="⚠️ Sistema em aquecimento. Tente novamente em 10 segundos.",
                sucesso=False,
            )

        t0     = time.time()
        config = {"configurable": {"session_id": state.session_id}}
        texto  = state.prompt_enriquecido or state.mensagem_original

        try:
            resultado = self._agent_with_history.invoke({"input": texto}, config=config)
            output    = resultado.get("output", "")

            validation = validar(state, output)
            latencia   = int((time.time() - t0) * 1000)

            obs.registrar_resposta(
                user_id=state.user_id,
                rota=state.rota.value,
                tokens_entrada=state.tokens_entrada,
                tokens_saida=state.tokens_saida,
                latencia_ms=latencia,
                iteracoes=state.iteracao_atual,
            )

            return AgentResponse(
                conteudo=validation.output,
                rota=state.rota,
                latencia_ms=latencia,
                sucesso=validation.valido,
            )

        except Exception as e:
            return self._tratar_erro(e, state, t0)

    def _tratar_erro(self, e: Exception, state: AgentState, t0: float) -> AgentResponse:
        err     = str(e)
        latencia = int((time.time() - t0) * 1000)

        # Rate limit 429
        if "429" in err or "rate_limit" in err.lower() or "too many requests" in err.lower():
            obs.warn(state.user_id, "Rate limit Groq", err[:200])
            return AgentResponse(conteudo=MSG_RATE_LIMIT, rota=state.rota,
                                 latencia_ms=latencia, sucesso=False)

        # tool_use_failed → limpa histórico corrompido e tenta sem ele
        if "400" in err and "tool_use_failed" in err:
            obs.warn(state.user_id, "tool_use_failed — limpando histórico", err[:200])
            limpar_historico(state.session_id)
            try:
                resultado = self._agent_executor.invoke(
                    {"input": state.mensagem_original, "history": []}
                )
                output     = resultado.get("output", "")
                validation = validar(state, output)
                return AgentResponse(
                    conteudo=validation.output,
                    rota=state.rota,
                    latencia_ms=int((time.time() - t0) * 1000),
                    sucesso=validation.valido,
                )
            except Exception as e2:
                obs.error(state.user_id, "Fallback falhou", str(e2)[:200])
                return AgentResponse(conteudo=MSG_HISTORICO_RESETADO, rota=state.rota,
                                     latencia_ms=latencia, sucesso=False)

        obs.error(state.user_id, "Erro crítico", err[:300])
        return AgentResponse(conteudo=MSG_ERRO_TECNICO, rota=state.rota,
                             latencia_ms=latencia, sucesso=False)


# Singleton
agent_core = AgentCore()