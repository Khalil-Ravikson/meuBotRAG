"""
agent/core.py — Orquestrador Principal (v3 — Clean Architecture)
=================================================================

O QUE MUDOU vs v2 (AgentExecutor LangChain + Groq):
─────────────────────────────────────────────────────
  ANTES:
    Mensagem → AgentExecutor (LangChain) → Groq (llama-3.1-8b)
                    ↓ tool_calling loop (até 6 iterações)
               pgvector → resposta

    Custo por mensagem: ~4.300 tokens, 500-1500ms, risco de rate limit Groq

  AGORA:
    Mensagem → Working Memory + Long-Term Facts (Redis, 0 tokens)
                    ↓
               Semantic Router (Redis KNN, 0 tokens, ~1ms)
                    ↓
               Query Transform (Gemini, ~120 tokens, 1 chamada leve)
                    ↓
               Hybrid Retriever (Redis BM25+Vetor, 0 tokens, ~5ms)
                    ↓
               Gemini Flash (contexto preciso, ~950 tokens, 1 chamada)
                    ↓
               Resposta → [background] Memory Extractor

    Custo por mensagem: ~1.070 tokens, 800-1200ms, free tier Gemini (1M TPM)

PIPELINE DETALHADA (por ordem de execução):
────────────────────────────────────────────
  1. MENU CHECK         → domain/menu.py (regex puro, 0ms, 0 tokens)
     Se for navegação de menu → retorna texto diretamente, FIM.

  2. WORKING MEMORY     → memory/working_memory.py (Redis, <1ms)
     Carrega histórico compactado da sessão atual (sliding window 8 turns)
     Carrega sinais: última tool, rota, tópico

  3. LONG-TERM FACTS    → memory/long_term_memory.py (Redis KNN, ~3ms)
     Busca fatos do utilizador relevantes para a pergunta atual
     Ex: "Aluno de Engenharia Civil, turno noturno"

  4. SEMANTIC ROUTING   → domain/semantic_router.py (Redis KNN, ~1ms)
     Determina qual tool/área responder SEM chamar LLM
     Alta confiança (>0.80): vai direto ao hybrid_retriever com source_filter
     Média confiança (0.62-0.80): usa retriever sem filtro
     Baixa confiança (<0.62): passa para Rota.GERAL sem retriever

  5. QUERY TRANSFORM    → rag/query_transform.py (Gemini, ~120 tokens)
     Se a query não for técnica: reescreve com contexto dos fatos
     Heurística _precisa_transformar() evita chamadas desnecessárias

  6. HYBRID RETRIEVAL   → rag/hybrid_retriever.py (Redis, ~5ms)
     BM25 + Vetor → RRF → contexto formatado com metadados hierárquicos
     Se 0 resultados: fallback step-back automático

  7. GERAÇÃO FINAL      → providers/gemini_provider.py (Gemini, ~950 tokens)
     Prompt = system + fatos + histórico compactado + contexto RAG + pergunta
     Uma única chamada limpa, sem tool-calling loop

  8. PERSISTÊNCIA       → working_memory.py (Redis, <1ms)
     Salva a pergunta e resposta no histórico da sessão

  9. EXTRAÇÃO [BG]      → memory/memory_extractor.py (async, não bloqueia)
     Analisa o turn e extrai novos fatos para a long-term memory
     Executado em background — não atrasa a resposta ao utilizador

MAPA DE FICHEIROS (onde cada responsabilidade vive):
─────────────────────────────────────────────────────
  Menu/navegação:     domain/menu.py          (não muda)
  Estado menu Redis:  memory/redis_memory.py  (não muda)
  Roteamento:         domain/semantic_router.py (novo)
  Memória de sessão:  memory/working_memory.py  (novo)
  Fatos long-term:    memory/long_term_memory.py (novo)
  Transform queries:  rag/query_transform.py    (novo)
  Busca híbrida:      rag/hybrid_retriever.py   (novo)
  Geração Gemini:     providers/gemini_provider.py (novo)
  Extração fatos:     memory/memory_extractor.py   (novo)
  Orquestração:       agent/core.py              (ESTE FICHEIRO)
  Envio WhatsApp:     services/evolution_service.py (não muda)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from src.domain.entities import AgentResponse, EstadoMenu, Rota
from src.domain.semantic_router import ResultadoRoteamento, rotear
from src.infrastructure.observability import obs
from src.infrastructure.settings import settings
from src.memory.long_term_memory import (
    Fato,
    buscar_fatos_relevantes,
    fatos_como_string,
)
from src.memory.working_memory import (
    HistoricoCompactado,
    adicionar_mensagem,
    get_historico_compactado,
    get_sinais,
    set_sinal,
)
from src.providers.gemini_provider import (
    SYSTEM_UEMA,
    GeminiResponse,
    chamar_gemini,
    montar_prompt_geracao,
)
from src.rag.hybrid_retriever import (
    ResultadoRecuperacao,
    recuperar,
    recuperar_simples,
)
from src.rag.query_transform import QueryTransformada, transformar_query

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mapeamento rota → source_filter e doc_type para o retriever
# ─────────────────────────────────────────────────────────────────────────────

_ROTA_PARA_SOURCE: dict[Rota, str] = {
    Rota.CALENDARIO: "calendario-academico-2026.pdf",
    Rota.EDITAL:     "edital_paes_2026.pdf",
    Rota.CONTATOS:   "guia_contatos_2025.pdf",
}

_ROTA_PARA_DOCTYPE: dict[Rota, str] = {
    Rota.CALENDARIO: "calendario",
    Rota.EDITAL:     "edital",
    Rota.CONTATOS:   "contatos",
}

# Mensagens de erro amigáveis (mantidas do prompts.py original)
_MSG_RATE_LIMIT    = "O sistema está com alta demanda. Aguarde alguns segundos e tente novamente. 🙏"
_MSG_SEM_INFO      = "Não consegui encontrar essa informação no momento. Tente reformular ou acesse uema.br."
_MSG_ERRO_TECNICO  = "Desculpe, tive uma dificuldade técnica. Tente novamente."
_MSG_AQUECENDO     = "⚠️ Sistema em aquecimento. Tente novamente em 10 segundos."


# ─────────────────────────────────────────────────────────────────────────────
# AgentCore
# ─────────────────────────────────────────────────────────────────────────────

class AgentCore:
    """
    Orquestrador principal do bot UEMA.

    É um singleton inicializado no startup do main.py.
    Não mantém estado interno entre chamadas — todo estado vive no Redis.
    Thread-safe por design (sem atributos mutáveis partilhados).
    """

    def __init__(self):
        self._inicializado = False
        self._tools: list = []

    def inicializar(self, tools: list) -> None:
        """
        Inicializa o agente com as tools disponíveis.
        Regista as tools no Redis para o Semantic Router.
        Chamado no startup do main.py após inicializar_indices().
        """
        self._tools = tools

        # Regista as tools no Redis para roteamento semântico
        try:
            from src.domain.semantic_router import registar_tools
            registar_tools(tools)
        except Exception as e:
            logger.warning("⚠️  Falha ao registar tools no router semântico: %s", e)

        self._inicializado = True
        logger.info("✅ AgentCore (v3) inicializado com %d tools.", len(tools))

    # ─────────────────────────────────────────────────────────────────────────
    # Ponto de entrada principal
    # ─────────────────────────────────────────────────────────────────────────

    def responder(
        self,
        user_id: str,
        session_id: str,
        mensagem: str,
        estado_menu: EstadoMenu = EstadoMenu.MAIN,
    ) -> AgentResponse:
        """
        Executa a pipeline completa e retorna a resposta.

        Este método é SÍNCRONO para compatibilidade com o código existente
        em handle_message.py que usa asyncio.to_thread().

        Parâmetros:
          user_id:    ID WhatsApp do utilizador (ex: "5598999999999")
          session_id: ID da sessão (igual ao user_id na maioria dos casos)
          mensagem:   Texto recebido do utilizador
          estado_menu: Estado atual do menu (para forçar rota em submenus)
        """
        if not self._inicializado:
            logger.warning("⚠️  AgentCore não inicializado.")
            return AgentResponse(conteudo=_MSG_AQUECENDO, sucesso=False)

        t0 = time.monotonic()

        try:
            resposta = self._executar_pipeline(user_id, session_id, mensagem, estado_menu)
        except Exception as e:
            logger.exception("❌ Erro não tratado na pipeline [%s]: %s", user_id, e)
            obs.error(user_id, "pipeline_erro_critico", str(e)[:300])
            resposta = AgentResponse(conteudo=_MSG_ERRO_TECNICO, sucesso=False)

        latencia_ms = int((time.monotonic() - t0) * 1000)
        resposta.latencia_ms = latencia_ms

        logger.info(
            "📤 [%s] latência=%dms | sucesso=%s | rota=%s",
            user_id, latencia_ms, resposta.sucesso, resposta.rota.value,
        )

        # Regista métricas de observabilidade
        obs.registrar_resposta(
            user_id=user_id,
            rota=resposta.rota.value,
            tokens_entrada=resposta.tokens_entrada,
            tokens_saida=resposta.tokens_saida,
            latencia_ms=latencia_ms,
            iteracoes=1,
        )

        return resposta

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline interna
    # ─────────────────────────────────────────────────────────────────────────

    def _executar_pipeline(
        self,
        user_id: str,
        session_id: str,
        mensagem: str,
        estado_menu: EstadoMenu,
    ) -> AgentResponse:
        """
        Os 9 passos da pipeline, em ordem.
        Cada passo pode fazer short-circuit e retornar diretamente.
        """

        # ── PASSO 1: Carregar Working Memory (Redis, <1ms) ────────────────────
        historico = get_historico_compactado(session_id)
        sinais    = get_sinais(session_id)

        logger.debug(
            "💭 Working memory [%s]: %d turns, %d chars",
            session_id, historico.turns_incluidos, historico.total_chars,
        )

        # ── PASSO 2: Carregar Fatos Long-Term (Redis KNN, ~3ms) ───────────────
        fatos = buscar_fatos_relevantes(user_id=user_id, pergunta=mensagem)

        if fatos:
            logger.debug(
                "🧠 Fatos relevantes [%s]: %d | top='%s'",
                user_id, len(fatos), fatos[0].texto[:50],
            )

        # ── PASSO 3: Semantic Routing (Redis KNN, ~1ms, 0 tokens) ────────────
        resultado_routing = rotear(mensagem, estado_menu)

        logger.info(
            "🗺️  Routing [%s]: rota=%s | confiança=%s | score=%.3f | método=%s",
            user_id,
            resultado_routing.rota.value,
            resultado_routing.confianca,
            resultado_routing.score,
            resultado_routing.metodo,
        )

        set_sinal(session_id, "rota",              resultado_routing.rota.value)
        set_sinal(session_id, "confianca_routing", resultado_routing.confianca)
        if resultado_routing.tool_name:
            set_sinal(session_id, "tool_usada", resultado_routing.tool_name)

        rota = resultado_routing.rota

        # ── PASSO 4: Query Transform (Gemini leve, ~120 tokens) ────────────
        # Alta confiança com rota específica → query já boa o suficiente
        usar_transform = not (
            resultado_routing.confianca == "alta"
            and rota != Rota.GERAL
        )

        if usar_transform:
            query_transformada = transformar_query(
                pergunta=mensagem,
                fatos_usuario=fatos,
                usar_sub_queries=(len(mensagem) > 80),
            )
        else:
            # Alta confiança: usa a mensagem directamente, sem custo Gemini
            from src.rag.query_transform import QueryTransformada
            query_transformada = QueryTransformada(
                query_original=mensagem,
                query_principal=mensagem,
                foi_transformada=False,
                motivo="alta confiança routing",
            )

        logger.debug("🔄 Query: %s", query_transformada.query_para_log)

        # ── PASSO 5: Hybrid Retrieval (Redis, ~5ms, 0 tokens) ──────────────
        recuperacao = self._executar_retrieval(rota, resultado_routing, query_transformada)

        # ── PASSO 6: Geração com Gemini (1 chamada limpa) ──────────────────
        gemini_resp = self._gerar_resposta(
            mensagem=mensagem,
            recuperacao=recuperacao,
            fatos=fatos,
            historico=historico,
            sinais=sinais,
            rota=rota,
        )

        if not gemini_resp.sucesso:
            return self._tratar_erro_gemini(gemini_resp, rota)

        conteudo = gemini_resp.conteudo.strip()
        if not conteudo or len(conteudo) < 10:
            return AgentResponse(
                conteudo=_MSG_SEM_INFO,
                rota=rota,
                sucesso=False,
            )

        # ── PASSO 7: Salva na Working Memory (Redis, <1ms) ─────────────────
        adicionar_mensagem(session_id, "user",      mensagem)
        adicionar_mensagem(session_id, "assistant", conteudo)
        set_sinal(session_id, "ultimo_topico", query_transformada.query_principal[:80])

        # ── PASSO 8: Extração de Fatos em Background ────────────────────────
        # Não bloqueia a resposta — lança tarefa assíncrona em background
        self._lancar_extracao_background(user_id, session_id)

        return AgentResponse(
            conteudo=conteudo,
            rota=rota,
            tokens_entrada=gemini_resp.input_tokens,
            tokens_saida=gemini_resp.output_tokens,
            sucesso=True,
        )

    def _executar_retrieval(
        self,
        rota: Rota,
        resultado_routing: ResultadoRoteamento,
        query_transformada: QueryTransformada,
    ) -> ResultadoRecuperacao:
        """
        Executa a busca híbrida com filtros baseados na rota.

        ESTRATÉGIA:
          - Rota específica (CALENDARIO/EDITAL/CONTATOS):
            → Filtra por source_filter (PDF exato) para maior precisão
          - Alta confiança routing:
            → Passa source_filter E doc_type para máxima precisão
          - Rota GERAL:
            → Busca sem filtros (pode trazer de qualquer PDF)
        """
        if rota == Rota.GERAL:
            # Sem filtro — busca em todos os documentos
            return recuperar(query_transformada)

        source_filter = _ROTA_PARA_SOURCE.get(rota)
        doc_type      = _ROTA_PARA_DOCTYPE.get(rota)

        # Alta confiança: filtro forte (source exato)
        if resultado_routing.confianca == "alta" and source_filter:
            return recuperar(
                query_transformada,
                source_filter=source_filter,
                doc_type=doc_type,
            )

        # Confiança média: filtro apenas por doc_type (mais flexível)
        return recuperar(query_transformada, doc_type=doc_type)

    def _gerar_resposta(
        self,
        mensagem: str,
        recuperacao: ResultadoRecuperacao,
        fatos: list[Fato],
        historico: HistoricoCompactado,
        sinais: dict,
        rota: Rota,
    ) -> GeminiResponse:
        """
        Monta o prompt final e chama o Gemini.

        ESTRUTURA DO PROMPT (por ordem, ~950 tokens totais):
        ─────────────────────────────────────────────────────
          [SYSTEM]         ~200 tokens  (SYSTEM_UEMA)
          [HISTÓRICO]      ~300 tokens  (sliding window compactado)
          [FATOS ALUNO]    ~100 tokens  (fatos relevantes da long-term)
          [CONTEXTO RAG]   ~600 tokens  (resultado híbrido formatado)
          [PERGUNTA]       ~50 tokens   (mensagem original)
          ─────────────────────────────
          TOTAL entrada:   ~1.250 tokens  (vs 4.300 no sistema antigo)
          Saída esperada:  ~200 tokens
        """
        # Monta contexto: histórico + pergunta nova
        historico_str = historico.texto_formatado
        working_memory_dict = {
            "ultimo_topico": sinais.get("ultimo_topico", ""),
            "tool_usada":    sinais.get("tool_usada", ""),
        }

        # Prompt do utilizador (contexto RAG + fatos + histórico)
        prompt_usuario = montar_prompt_geracao(
            pergunta=mensagem,
            contexto_rag=recuperacao.contexto_formatado,
            working_memory=working_memory_dict if any(working_memory_dict.values()) else None,
            fatos_usuario=[f.texto for f in fatos] if fatos else None,
        )

        # Adiciona histórico ao system instruction para manter contexto de conversa
        system_com_historico = SYSTEM_UEMA
        if historico_str:
            system_com_historico = (
                f"{SYSTEM_UEMA}\n\n"
                f"[HISTÓRICO RECENTE DA CONVERSA]\n{historico_str}"
            )

        return chamar_gemini(
            prompt=prompt_usuario,
            system_instruction=system_com_historico,
            temperatura=settings.GEMINI_TEMP,
            max_tokens=settings.GEMINI_MAX_TOKENS,
        )

    def _tratar_erro_gemini(
        self, resp: GeminiResponse, rota: Rota
    ) -> AgentResponse:
        """Traduz erros do Gemini em mensagens amigáveis."""
        erro = resp.erro.lower()

        if "429" in erro or "quota" in erro or "rate" in erro:
            obs.warn("SYSTEM", "gemini_rate_limit", resp.erro[:200])
            return AgentResponse(conteudo=_MSG_RATE_LIMIT, rota=rota, sucesso=False)

        if "503" in erro or "overloaded" in erro:
            return AgentResponse(conteudo=_MSG_RATE_LIMIT, rota=rota, sucesso=False)

        obs.error("SYSTEM", "gemini_erro", resp.erro[:200])
        return AgentResponse(conteudo=_MSG_ERRO_TECNICO, rota=rota, sucesso=False)

    def _lancar_extracao_background(self, user_id: str, session_id: str) -> None:
        """
        Lança a extração de fatos em background sem bloquear a resposta.

        USA asyncio.get_event_loop().call_soon_threadsafe() para ser seguro
        mesmo quando chamado de dentro de asyncio.to_thread().

        Se o event loop não estiver disponível (ex: testes), executa
        síncronamente com supressão de erros.
        """
        try:
            from src.memory.memory_extractor import extrair_fatos_do_ultimo_turn
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Estamos dentro do event loop do FastAPI → cria task
                asyncio.ensure_future(
                    asyncio.to_thread(extrair_fatos_do_ultimo_turn, user_id, session_id)
                )
            else:
                # Fora do event loop (ex: testes, Chainlit) → executa direto
                extrair_fatos_do_ultimo_turn(user_id, session_id)
        except Exception as e:
            # Extração em background nunca deve quebrar o fluxo principal
            logger.debug("ℹ️  Extração background ignorada: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton global (compatível com o código existente)
# ─────────────────────────────────────────────────────────────────────────────

agent_core = AgentCore()