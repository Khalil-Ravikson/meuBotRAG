"""
agent/core.py — Orquestrador Principal (v4 — Cache + Wiki)
===========================================================

MUDANÇAS v4 vs v3:
───────────────────
  1. Semantic Cache integrado (Passo 5.5 e 7.5):
       - Passo 5.5: check_cache() antes de chamar Gemini
       - Passo 7.5: store_cache() após geração bem-sucedida
       - Cache hit → resposta em ~5ms, 0 tokens Gemini

  2. Rota.WIKI adicionada:
       - _ROTA_PARA_SOURCE: Rota.WIKI → source_filter dinâmico (wiki:*)
       - Busca híbrida sem source_filter fixo (multi-página Wiki)

  3. Correcção do import vector_store → embeddings (indirecto via long_term_memory)

PIPELINE v4 (10 passos):
──────────────────────────
  1.  Working Memory       (Redis, <1ms)
  2.  Long-Term Facts      (Redis KNN, ~3ms) ← BUG CORRIGIDO: import embeddings
  3.  Semantic Routing     (Redis KNN, ~1ms, 0 tokens)
  4.  Query Transform      (Gemini, ~120 tokens — apenas quando necessário)
  5.  Hybrid Retrieval     (Redis, ~5ms, 0 tokens)
  5.5 Semantic Cache CHECK (Redis, ~3ms, 0 tokens) ← NOVO v4
  6.  Gemini Generation    (~950 tokens — só se cache miss)
  7.  Persist Memory       (Redis, <1ms)
  7.5 Semantic Cache STORE (Redis, ~3ms) ← NOVO v4
  8.  Background Extractor (daemon thread, não bloqueia)
"""
from __future__ import annotations

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field

from src.domain.entities import AgentResponse, EstadoMenu, Rota
from src.domain.semantic_router import ResultadoRoteamento, rotear
from src.infrastructure.observability import obs
from src.infrastructure.settings import settings
from src.memory.long_term_memory import Fato, buscar_fatos_relevantes, fatos_como_string
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
from src.rag.hybrid_retriever import ResultadoRecuperacao, recuperar, recuperar_simples
from src.rag.query_transform import QueryTransformada, transformar_query

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mapeamentos rota → source
# ─────────────────────────────────────────────────────────────────────────────

_ROTA_PARA_SOURCE: dict[Rota, str] = {
    Rota.CALENDARIO: "calendario-academico-2026.pdf",
    Rota.EDITAL:     "edital_paes_2026.pdf",
    Rota.CONTATOS:   "guia_contatos_2025.pdf",
    Rota.WIKI:       None,   # ← Wiki: sem source único, busca em wiki:*
}

_ROTA_PARA_DOCTYPE: dict[Rota, str] = {
    Rota.CALENDARIO: "calendario",
    Rota.EDITAL:     "edital",
    Rota.CONTATOS:   "contatos",
    Rota.WIKI:       "wiki_ctic",
}

_MSG_RATE_LIMIT   = "O sistema está com alta demanda. Aguarde alguns segundos e tente novamente. 🙏"
_MSG_SEM_INFO     = "Não consegui encontrar essa informação no momento. Tente reformular ou acesse uema.br."
_MSG_ERRO_TECNICO = "Desculpe, tive uma dificuldade técnica. Tente novamente."
_MSG_AQUECENDO    = "⚠️ Sistema em aquecimento. Tente novamente em 10 segundos."


# ─────────────────────────────────────────────────────────────────────────────
# AgentCore
# ─────────────────────────────────────────────────────────────────────────────

class AgentCore:
    """
    Orquestrador principal do bot UEMA.
    Singleton. Todo estado vive no Redis — thread-safe por design.
    """

    def __init__(self):
        self._inicializado = False
        self._tools: list  = []

    def inicializar(self, tools: list) -> None:
        self._tools = tools
        try:
            from src.domain.semantic_router import registar_tools
            registar_tools(tools)
        except Exception as e:
            logger.warning("⚠️  Falha ao registar tools no router semântico: %s", e)

        # Inicializa índice do Semantic Cache
        try:
            from src.infrastructure.semantic_cache import init_cache_index
            init_cache_index()
        except Exception as e:
            logger.warning("⚠️  Falha ao inicializar Semantic Cache: %s", e)

        self._inicializado = True
        logger.info("✅ AgentCore (v4) inicializado com %d tools.", len(tools))

    # ─────────────────────────────────────────────────────────────────────────
    # Ponto de entrada
    # ─────────────────────────────────────────────────────────────────────────

    def responder(
        self,
        user_id:     str,
        session_id:  str,
        mensagem:    str,
        estado_menu: EstadoMenu = EstadoMenu.MAIN,
    ) -> AgentResponse:
        if not self._inicializado:
            return AgentResponse(conteudo=_MSG_AQUECENDO, sucesso=False)

        t0 = time.monotonic()
        try:
            resposta = self._executar_pipeline(user_id, session_id, mensagem, estado_menu)
        except Exception as e:
            logger.exception("❌ Pipeline crítica [%s]: %s", user_id, e)
            obs.error(user_id, "pipeline_erro_critico", str(e)[:300])
            resposta = AgentResponse(conteudo=_MSG_ERRO_TECNICO, sucesso=False)

        resposta.latencia_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "📤 [%s] latência=%dms | sucesso=%s | rota=%s",
            user_id, resposta.latencia_ms, resposta.sucesso, resposta.rota.value,
        )
        obs.registrar_resposta(
            user_id=user_id, rota=resposta.rota.value,
            tokens_entrada=resposta.tokens_entrada, tokens_saida=resposta.tokens_saida,
            latencia_ms=resposta.latencia_ms, iteracoes=1,
        )
        return resposta

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline interna
    # ─────────────────────────────────────────────────────────────────────────

    def _executar_pipeline(
        self,
        user_id:     str,
        session_id:  str,
        mensagem:    str,
        estado_menu: EstadoMenu,
    ) -> AgentResponse:

        # ── PASSO 1: Working Memory ───────────────────────────────────────────
        historico = get_historico_compactado(session_id)
        sinais    = get_sinais(session_id)

        # ── PASSO 2: Long-Term Facts (BUG CORRIGIDO: import embeddings) ───────
        fatos = buscar_fatos_relevantes(user_id=user_id, pergunta=mensagem)
        if fatos:
            logger.debug("🧠 Fatos [%s]: %d | top='%s'", user_id, len(fatos), fatos[0].texto[:50])

        # ── PASSO 3: Semantic Routing ─────────────────────────────────────────
        resultado_routing = rotear(mensagem, estado_menu)
        rota              = resultado_routing.rota
        logger.info(
            "🗺️  Routing [%s]: rota=%s | confiança=%s | score=%.3f | método=%s",
            user_id, rota.value, resultado_routing.confianca,
            resultado_routing.score, resultado_routing.metodo,
        )
        set_sinal(session_id, "rota",              rota.value)
        set_sinal(session_id, "confianca_routing", resultado_routing.confianca)
        if resultado_routing.tool_name:
            set_sinal(session_id, "tool_usada", resultado_routing.tool_name)

        # ── PASSO 4: Query Transform ──────────────────────────────────────────
        usar_transform = not (resultado_routing.confianca == "alta" and rota != Rota.GERAL)
        if usar_transform:
            query_transformada = transformar_query(
                pergunta=mensagem,
                fatos_usuario=fatos,
                usar_sub_queries=(len(mensagem) > 80),
            )
        else:
            query_transformada = QueryTransformada(
                query_original=mensagem,
                query_principal=mensagem,
                foi_transformada=False,
                motivo="alta confiança routing",
            )
        logger.debug("🔄 Query: %s", query_transformada.query_para_log)

        # ── PASSO 5: Hybrid Retrieval ─────────────────────────────────────────
        recuperacao = self._executar_retrieval(rota, resultado_routing, query_transformada)

        # ── PASSO 5.5: Semantic Cache CHECK (NOVO v4) ─────────────────────────
        # Apenas para rotas com RAG — não cacheia respostas genéricas
        if rota != Rota.GERAL:
            try:
                from src.infrastructure.semantic_cache import check_cache
                cache_hit = check_cache(
                    query    = query_transformada.query_principal,
                    doc_type = rota.value,
                )
                if cache_hit:
                    adicionar_mensagem(session_id, "user",      mensagem)
                    adicionar_mensagem(session_id, "assistant", cache_hit)
                    logger.info("🎯 Cache HIT [%s] rota=%s", user_id, rota.value)
                    return AgentResponse(
                        conteudo       = cache_hit,
                        rota           = rota,
                        tokens_entrada = 0,
                        tokens_saida   = 0,
                        sucesso        = True,
                    )
            except Exception as e:
                logger.debug("ℹ️  Cache check ignorado: %s", e)

        # ── PASSO 6: Geração com Gemini ───────────────────────────────────────
        gemini_resp = self._gerar_resposta(
            mensagem=mensagem, recuperacao=recuperacao,
            fatos=fatos, historico=historico,
        )

        conteudo = gemini_resp.conteudo if gemini_resp.sucesso else _MSG_ERRO_TECNICO
        if "RATE_LIMIT" in conteudo.upper():
            conteudo = _MSG_RATE_LIMIT

        # ── PASSO 7: Persistência ─────────────────────────────────────────────
        adicionar_mensagem(session_id, "user",      mensagem)
        adicionar_mensagem(session_id, "assistant", conteudo)

        # ── PASSO 7.5: Semantic Cache STORE (NOVO v4) ─────────────────────────
        if gemini_resp.sucesso and rota != Rota.GERAL and len(conteudo) > 50:
            try:
                from src.infrastructure.semantic_cache import store_cache
                store_cache(
                    query    = query_transformada.query_principal,
                    response = conteudo,
                    doc_type = rota.value,
                )
            except Exception as e:
                logger.debug("ℹ️  Cache store ignorado: %s", e)

        # ── PASSO 8: Background Extractor ─────────────────────────────────────
        self._extrair_fatos_background(user_id, session_id)

        return AgentResponse(
            conteudo       = conteudo,
            rota           = rota,
            tokens_entrada = gemini_resp.tokens_entrada,
            tokens_saida   = gemini_resp.tokens_saida,
            sucesso        = gemini_resp.sucesso,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def _executar_retrieval(
        self,
        rota:               Rota,
        resultado_routing:  ResultadoRoteamento,
        query_transformada: QueryTransformada,
    ) -> ResultadoRecuperacao:
        if rota == Rota.GERAL:
            return ResultadoRecuperacao(encontrou=False, metodo_usado="geral_sem_rag")

        source_filter = _ROTA_PARA_SOURCE.get(rota)

        # Wiki: busca por doc_type em vez de source fixo
        if rota == Rota.WIKI:
            return recuperar(
                query          = query_transformada.query_principal,
                source_filter  = None,
                doc_type_filter= "wiki_ctic",
            )

        recuperacao = recuperar(
            query          = query_transformada.query_principal,
            source_filter  = source_filter,
        )

        # Fallback step-back: se não encontrou, tenta query mais ampla
        if not recuperacao.encontrou and query_transformada.foi_transformada:
            logger.debug("🔁 Step-back fallback para query original.")
            recuperacao = recuperar_simples(query_transformada.query_original)

        return recuperacao

    # ─────────────────────────────────────────────────────────────────────────
    # Geração
    # ─────────────────────────────────────────────────────────────────────────

    def _gerar_resposta(
        self,
        mensagem:    str,
        recuperacao: ResultadoRecuperacao,
        fatos:       list[Fato],
        historico:   HistoricoCompactado,
    ) -> GeminiResponse:
        prompt = montar_prompt_geracao(
            pergunta         = mensagem,
            contexto_rag     = recuperacao.contexto_formatado if recuperacao.encontrou else "",
            fatos_usuario    = fatos_como_string(fatos),
            historico        = historico.texto_formatado,
        )
        return chamar_gemini(system=SYSTEM_UEMA, user_prompt=prompt)

    # ─────────────────────────────────────────────────────────────────────────
    # Background extractor
    # ─────────────────────────────────────────────────────────────────────────

    def _extrair_fatos_background(self, user_id: str, session_id: str) -> None:
        from src.memory.memory_extractor import extrair_fatos_do_ultimo_turn

        def _run():
            try:
                extrair_fatos_do_ultimo_turn(user_id, session_id)
            except Exception as e:
                logger.debug("ℹ️  Extração background ignorada [%s]: %s", user_id, e)

        try:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(asyncio.to_thread(extrair_fatos_do_ultimo_turn, user_id, session_id), loop=loop)
        except RuntimeError:
            t = threading.Thread(target=_run, daemon=True, name=f"extractor-{user_id[:8]}")
            t.start()
        except Exception as e:
            logger.debug("ℹ️  Extração background ignorada: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

agent_core = AgentCore()