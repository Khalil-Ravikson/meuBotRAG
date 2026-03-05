"""
agent/core.py — Orquestrador Principal (v3 — Clean Architecture)
=================================================================
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

# Imports atualizados (EstadoMenu removido)
from src.domain.guardrails import guardrails
from src.domain.entities import AgentResponse, Rota
from src.domain.semantic_router import ResultadoRoteamento, rotear
from src.infrastructure.observability import obs
from src.infrastructure.settings import settings
from src.infrastructure.semantic_cache import check_cache, store_cache

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

_MSG_RATE_LIMIT    = "O sistema está com alta demanda. Aguarde alguns segundos e tente novamente. 🙏"
_MSG_SEM_INFO      = "Não consegui encontrar essa informação no momento. Tente reformular ou acesse uema.br."
_MSG_ERRO_TECNICO  = "Desculpe, tive uma dificuldade técnica. Tente novamente."
_MSG_AQUECENDO     = "⚠️ Sistema em aquecimento. Tente novamente em 10 segundos."

# ─────────────────────────────────────────────────────────────────────────────
# AgentCore
# ─────────────────────────────────────────────────────────────────────────────

class AgentCore:
    def __init__(self):
        self._inicializado = False
        self._tools: list = []

    def inicializar(self, tools: list) -> None:
        self._tools = tools
        try:
            from src.domain.semantic_router import registar_tools
            registar_tools(tools)
        except Exception as e:
            logger.warning("⚠️  Falha ao registar tools no router semântico: %s", e)

        self._inicializado = True
        logger.info("✅ AgentCore (v3) inicializado com %d tools.", len(tools))

    def responder(
        self,
        user_id: str,
        session_id: str,
        mensagem: str,
    ) -> AgentResponse:
        
        if not self._inicializado:
            logger.warning("⚠️  AgentCore não inicializado.")
            return AgentResponse(conteudo=_MSG_AQUECENDO, sucesso=False)
        
        # ── GUARDRAIL CHECK ────────────────────────────────────────────────────
        guardrail = guardrails.analisar(mensagem)
        if guardrail.bloquear:
            return AgentResponse(
                conteudo=guardrail.resposta,
                rota=Rota.GERAL,
                sucesso=True,
                tokens_entrada=0,
                tokens_saida=0
            )
        # ────────────────────────────────────────────────────────────────────────

        t0 = time.monotonic()

        try:
            # Passamos o sinal 'precisa_rag' do guardrail para a pipeline
            resposta = self._executar_pipeline(user_id, session_id, mensagem, guardrail.precisa_rag)
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

        obs.registrar_resposta(
            user_id=user_id,
            rota=resposta.rota.value,
            tokens_entrada=resposta.tokens_entrada,
            tokens_saida=resposta.tokens_saida,
            latencia_ms=latencia_ms,
            iteracoes=1,
        )

        return resposta

    def _executar_pipeline(
        self,
        user_id: str,
        session_id: str,
        mensagem: str,
        precisa_rag: bool,
    ) -> AgentResponse:

        # ── PASSO 1: Working Memory ──────────────────────────────────────────
        historico = get_historico_compactado(session_id)
        sinais    = get_sinais(session_id)

        # ── PASSO 2: Long-Term Facts ─────────────────────────────────────────
        fatos = buscar_fatos_relevantes(user_id=user_id, pergunta=mensagem)

        # ── PASSO 3: Semantic Routing ────────────────────────────────────────
        resultado_routing = rotear(mensagem)  # Estado menu removido
        rota = resultado_routing.rota

        set_sinal(session_id, "rota", rota.value)
        set_sinal(session_id, "confianca_routing", resultado_routing.confianca)

        # ── PASSO 4: Query Transform ─────────────────────────────────────────
        usar_transform = not (resultado_routing.confianca == "alta" and rota != Rota.GERAL)

        if usar_transform:
            query_transformada = transformar_query(mensagem, fatos, (len(mensagem) > 80))
        else:
            query_transformada = QueryTransformada(
                query_original=mensagem,
                query_principal=mensagem,
                foi_transformada=False,
                motivo="alta confiança routing",
            )

        # ── PASSO 5: Semantic Cache Check ────────────────────────────────────
        cache_hit = check_cache(
            query=query_transformada.query_principal,
            doc_type=rota.value,
        )
        
        if cache_hit:
            # Bypass total do Gemini se a resposta já existir
            adicionar_mensagem(session_id, "user", mensagem)
            adicionar_mensagem(session_id, "assistant", cache_hit)
            set_sinal(session_id, "ultimo_topico", query_transformada.query_principal[:80])
            self._lancar_extracao_background(user_id, session_id)
            
            return AgentResponse(
                conteudo=cache_hit, 
                rota=rota, 
                tokens_entrada=0, 
                tokens_saida=0, 
                sucesso=True
            )

        # ── PASSO 6: Hybrid Retrieval (com Self-RAG skip) ────────────────────
        usar_rag = precisa_rag and rota != Rota.GERAL
        
        if usar_rag:
            recuperacao = self._executar_retrieval(rota, resultado_routing, query_transformada)
        else:
            recuperacao = ResultadoRecuperacao(encontrou=False, metodo_usado="self_rag_skip", chunks=[], contexto_formatado="")

        # ── PASSO 7: Geração Final (Gemini) ──────────────────────────────────
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
            return AgentResponse(conteudo=_MSG_SEM_INFO, rota=rota, sucesso=False)

        # ── PASSO 8: Semantic Cache Store ────────────────────────────────────
        store_cache(
            query=query_transformada.query_principal,
            response=conteudo,
            doc_type=rota.value,
        )

        # ── PASSO 9: Persistência & Extração BG ──────────────────────────────
        adicionar_mensagem(session_id, "user", mensagem)
        adicionar_mensagem(session_id, "assistant", conteudo)
        set_sinal(session_id, "ultimo_topico", query_transformada.query_principal[:80])

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
        if rota == Rota.GERAL:
            return recuperar(query_transformada)

        source_filter = _ROTA_PARA_SOURCE.get(rota)
        doc_type      = _ROTA_PARA_DOCTYPE.get(rota)

        if resultado_routing.confianca == "alta" and source_filter:
            return recuperar(query_transformada, source_filter=source_filter, doc_type=doc_type)

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
        historico_str = historico.texto_formatado
        working_memory_dict = {
            "ultimo_topico": sinais.get("ultimo_topico", ""),
            "tool_usada":    sinais.get("tool_usada", ""),
        }

        prompt_usuario = montar_prompt_geracao(
            pergunta=mensagem,
            contexto_rag=recuperacao.contexto_formatado,
            working_memory=working_memory_dict if any(working_memory_dict.values()) else None,
            fatos_usuario=[f.texto for f in fatos] if fatos else None,
        )

        system_com_historico = SYSTEM_UEMA
        if historico_str:
            system_com_historico = f"{SYSTEM_UEMA}\n\n[HISTÓRICO RECENTE DA CONVERSA]\n{historico_str}"

        return chamar_gemini(
            prompt=prompt_usuario,
            system_instruction=system_com_historico,
            temperatura=settings.GEMINI_TEMP,
            max_tokens=settings.GEMINI_MAX_TOKENS,
        )

    def _tratar_erro_gemini(self, resp: GeminiResponse, rota: Rota) -> AgentResponse:
        erro = resp.erro.lower()
        if "429" in erro or "quota" in erro or "rate" in erro or "503" in erro or "overloaded" in erro:
            obs.warn("SYSTEM", "gemini_rate_limit", resp.erro[:200])
            return AgentResponse(conteudo=_MSG_RATE_LIMIT, rota=rota, sucesso=False)

        obs.error("SYSTEM", "gemini_erro", resp.erro[:200])
        return AgentResponse(conteudo=_MSG_ERRO_TECNICO, rota=rota, sucesso=False)

    def _lancar_extracao_background(self, user_id: str, session_id: str) -> None:
        import threading
        from src.memory.memory_extractor import extrair_fatos_do_ultimo_turn

        def _run_in_thread() -> None:
            try:
                extrair_fatos_do_ultimo_turn(user_id, session_id)
            except Exception as e:
                logger.debug("ℹ️  Extração background ignorada [%s]: %s", user_id, e)

        try:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(
                asyncio.to_thread(extrair_fatos_do_ultimo_turn, user_id, session_id),
                loop=loop,
            )
        except RuntimeError:
            t = threading.Thread(target=_run_in_thread, daemon=True, name=f"extractor-{user_id[:8]}")
            t.start()
        except Exception as e:
            logger.debug("ℹ️  Extração background ignorada: %s", e)

agent_core = AgentCore()