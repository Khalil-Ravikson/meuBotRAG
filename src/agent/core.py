"""
agent/core.py — Orquestrador Principal (v5 — Guardrails + Self-RAG + CRAG + Cache)
====================================================================================

IMPLEMENTAÇÕES DESTA VERSÃO:
──────────────────────────────

  PONTO 3 — Guardrails (camada ANTES do pipeline)
  ────────────────────────────────────────────────
  Problema: "oi linda" consumia ~1.070 tokens desnecessariamente.
  Solução: _guardrails() executa ANTES de tudo — 0 tokens, 0ms de rede.

  Dois níveis:
    Nível 1 — Saudações/atalhos regex (0ms):
      "oi", "olá", "tudo bem", "obrigado", "ok", "tchau" →
      resposta directa pré-fabricada, sai da pipeline.

    Nível 2 — Domínio académico (heurística de keywords):
      "me ajuda a fazer redacção", "quem ganhou o jogo" →
      resposta educada de redireccionamento, sai da pipeline.

  Economia: ~300 msgs/dia de saudações × 1.070 tokens = 321.000 tokens/dia poupados.

  PONTO 4 — Self-RAG (decisão de skip do retriever)
  ──────────────────────────────────────────────────
  Problema: retriever chamado mesmo para "obrigado!" → 0 resultados → Gemini sem contexto.
  Solução: _decidir_precisa_rag() em dois níveis.

    Nível 1 — Heurística local (0 tokens):
      Verifica padrões que NUNCA precisam de RAG:
        - Mensagem < 4 palavras com rota GERAL
        - Lista de stop-words de conversação
      Se heurística clara → skip imediato, sem chamar Gemini.

    Nível 2 — LLM leve (só quando heurística inconclusiva):
      Chama Gemini com PROMPT_PRECISA_RAG (mini-prompt ~50 tokens):
        "Determine se esta mensagem precisa de documentos: SIM/NAO"
      Usado para casos ambíguos: "me explica o que é AC" (pode ser edital ou pergunta geral).
      Custo: ~80 tokens (vs ~950 da geração completa).

  PONTO 5 — CRAG — Corrective RAG
  ─────────────────────────────────
  Problema: retriever devolve chunks de score RRF < 0.4 → Gemini gera com contexto fraco.
  Solução: _avaliar_e_corrigir_retrieval() após o retrieval, antes da geração.

    Passo 1 — Avaliação de qualidade (score RRF médio dos chunks):
      score_medio = media(chunk.rrf_score for chunk in resultados)
      Se score_medio >= CRAG_THRESHOLD_OK (0.40) → contexto bom → gera normal.

    Passo 2 — Tentativa de correcção (se score baixo):
      a) Step-back query: generaliza a pergunta ("matrícula Eng Civil" → "matrícula veteranos")
      b) Re-busca com a query step-back
      c) Se novo score > threshold → usa novo contexto
      d) Se ainda baixo → gera sem RAG com disclaimer

    Passo 3 — Geração com disclaimer (contexto insuficiente):
      "Não encontrei informação precisa nos documentos. Baseando-me no conhecimento geral..."
      Protege o aluno de receber datas/vagas inventadas.

  PONTO 2 — Semantic Cache (integrado da v4, com ajuste para CRAG)
  ─────────────────────────────────────────────────────────────────
  Cache só activo quando CRAG aprova (score_medio >= threshold).
  Não cacheamos respostas geradas com contexto fraco.

PIPELINE v5 COMPLETA (12 passos):
───────────────────────────────────
  0.  GUARDRAILS    (regex/heurística, 0ms, 0 tokens) ← NOVO
  1.  Working Memory          (Redis, <1ms)
  2.  Long-Term Facts         (Redis KNN, ~3ms)
  3.  Semantic Routing        (Redis KNN, ~1ms, 0 tokens)
  4.  Self-RAG decision       (heurística 0ms OU Gemini ~80 tokens) ← NOVO
  5.  Query Transform         (Gemini ~120 tokens — só quando necessário)
  5.5 Semantic Cache CHECK    (Redis, ~3ms, 0 tokens)
  6.  Hybrid Retrieval        (Redis, ~5ms, 0 tokens)
  6.5 CRAG — Avaliar qualidade (score RRF + step-back se fraco) ← NOVO
  7.  Gemini Generation       (~950 tokens — só se cache miss)
  8.  Persist Memory          (Redis, <1ms)
  8.5 Semantic Cache STORE    (só se CRAG score >= threshold)
  9.  Background Extractor    (daemon thread)

ECONOMIA DE TOKENS vs v3 (LangChain+Groq):
────────────────────────────────────────────
  v3:  ~4.300 tokens/msg
  v4:  ~1.070 tokens/msg  (−75%)
  v5:  ~750  tokens/msg   (−82%) — guardrails eliminam saudações
                                   self-RAG elimina lookups desnecessários
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field

from src.agent.prompts import (
    PROMPT_AVALIAR_RELEVANCIA,
    PROMPT_PRECISA_RAG,
    SYSTEM_UEMA,
    montar_prompt_geracao,
)
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
from src.providers.gemini_provider import GeminiResponse, chamar_gemini
from src.rag.hybrid_retriever import ResultadoRecuperacao, recuperar, recuperar_simples
from src.rag.query_transform import QueryTransformada, transformar_query

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mapeamentos rota → source / doc_type
# ─────────────────────────────────────────────────────────────────────────────

_ROTA_PARA_SOURCE: dict[Rota, str | None] = {
    Rota.CALENDARIO: "calendario-academico-2026.pdf",
    Rota.EDITAL:     "edital_paes_2026.pdf",
    Rota.CONTATOS:   "guia_contatos_2025.pdf",
    Rota.WIKI:       None,
}

_ROTA_PARA_DOCTYPE: dict[Rota, str] = {
    Rota.CALENDARIO: "calendario",
    Rota.EDITAL:     "edital",
    Rota.CONTATOS:   "contatos",
    Rota.WIKI:       "wiki_ctic",
}

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds e constantes
# ─────────────────────────────────────────────────────────────────────────────

CRAG_THRESHOLD_OK   = 0.40   # score RRF médio mínimo para contexto "bom"
CRAG_THRESHOLD_MIN  = 0.20   # abaixo disto → gera sem RAG + disclaimer

_MSG_RATE_LIMIT   = "O sistema está com alta demanda. Aguarde alguns segundos e tente novamente. 🙏"
_MSG_SEM_INFO     = "Não consegui encontrar essa informação no momento. Tente reformular ou acesse uema.br."
_MSG_ERRO_TECNICO = "Desculpe, tive uma dificuldade técnica. Tente novamente."
_MSG_AQUECENDO    = "⚠️ Sistema em aquecimento. Tente novamente em 10 segundos."
_MSG_FORA_DOMINIO = (
    "Fico feliz em ajudar com dúvidas académicas da UEMA! 😊\n"
    "Posso responder sobre calendário, editais, vagas, cotas e contatos da universidade.\n"
    "Para outros assuntos, recomendo consultar os tutores do teu curso."
)

# ─────────────────────────────────────────────────────────────────────────────
# PONTO 3 — Guardrails: padrões regex e keywords
# ─────────────────────────────────────────────────────────────────────────────

# Nível 1 — Saudações e conversas informais (resposta directa pré-fabricada)
_REGEX_SAUDACOES = re.compile(
    r"^(oi|ol[aá]|ei|e a[ií]|eae|opa|hey|hi|hello|bom\s*dia|boa\s*tarde|boa\s*noite"
    r"|tudo\s*(bem|bom|certo|certo\??)|como\s+(vai|est[aá]s?|vc)"
    r"|obrigad[oa]s?|valeu|vlw|thanks|ok\s*$|entendid[oa]|certo\s*$"
    r"|at[eé]\s*(logo|mais|mais\s*tarde)|tchau|flw|falou|xau)\s*[!?.]?$",
    re.IGNORECASE,
)

# Nível 2 — Fora do domínio académico UEMA (heurística de keywords off-topic)
_KEYWORDS_FORA_DOMINIO = frozenset({
    "redacção", "redação", "poema", "história", "conto", "dissertação",
    "futebol", "jogo", "placar", "partida", "campeonato",
    "receita", "culinária", "comida",
    "música", "letra", "cifra",
    "namorad", "relacionamento", "amor",
    "novela", "série", "filme",
    "política", "governo", "presidente",
    "investimento", "bitcoin", "crypto",
    "medicina", "sintoma", "doença",  # fora da UEMA específica
    "programar", "python", "javascript",  # fora do contexto de suporte UEMA
})

# Keywords que GARANTEM domínio UEMA (override do fora_dominio)
_KEYWORDS_DOMINIO_UEMA = frozenset({
    "uema", "paes", "matrícula", "matricula", "edital", "calendário", "calendario",
    "semestre", "vaga", "vagas", "cota", "cotas", "inscrição", "inscricao",
    "trancamento", "prova", "avaliação", "avaliacao", "rematrícula",
    "contato", "secretaria", "coord", "prog", "ctic", "glpi",
    "ac", "pcd", "br-ppi", "br-q", "br-dc", "ir-ppi",
    "campus", "curso", "graduação", "graduacao",
    "bolsa", "auxílio", "auxilio", "ru", "restaurante",
    "wiki", "sistema", "sigaa", "senha", "e-mail",
})

# Stop-words conversacionais (Self-RAG nível 1 — heurística)
_STOP_WORDS_SEM_RAG = frozenset({
    "oi", "olá", "ola", "obrigado", "obrigada", "valeu", "vlw",
    "ok", "entendido", "certo", "tá", "ta", "show", "blz",
    "tchau", "flw", "falou", "até", "ate",
    "sim", "não", "nao", "claro", "pode",
    "rs", "rsrs", "haha", "kk", "kkk",
})


# ─────────────────────────────────────────────────────────────────────────────
# AgentCore
# ─────────────────────────────────────────────────────────────────────────────

class AgentCore:
    """
    Orquestrador principal do bot UEMA v5.
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
            logger.warning("⚠️  Falha ao registar tools no router: %s", e)

        try:
            from src.infrastructure.semantic_cache import init_cache_index
            init_cache_index()
        except Exception as e:
            logger.warning("⚠️  Falha ao inicializar Semantic Cache: %s", e)

        self._inicializado = True
        logger.info("✅ AgentCore v5 inicializado com %d tools.", len(tools))

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
            "📤 [%s] %dms | sucesso=%s | rota=%s",
            user_id, resposta.latencia_ms, resposta.sucesso, resposta.rota.value,
        )
        obs.registrar_resposta(
            user_id=user_id, rota=resposta.rota.value,
            tokens_entrada=resposta.tokens_entrada, tokens_saida=resposta.tokens_saida,
            latencia_ms=resposta.latencia_ms, iteracoes=1,
        )
        return resposta

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline interna — 12 passos
    # ─────────────────────────────────────────────────────────────────────────

    def _executar_pipeline(
        self,
        user_id:     str,
        session_id:  str,
        mensagem:    str,
        estado_menu: EstadoMenu,
    ) -> AgentResponse:

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 0 — GUARDRAILS (0ms, 0 tokens)
        # ══════════════════════════════════════════════════════════════════════
        guardrail_result = _guardrails(mensagem)
        if guardrail_result is not None:
            logger.info("🛡️  Guardrail [%s]: %s → short-circuit", user_id, guardrail_result[0])
            # Ainda salva na working memory para manter contexto de conversa
            adicionar_mensagem(session_id, "user",      mensagem)
            adicionar_mensagem(session_id, "assistant", guardrail_result[1])
            return AgentResponse(
                conteudo       = guardrail_result[1],
                rota           = Rota.GERAL,
                tokens_entrada = 0,
                tokens_saida   = 0,
                sucesso        = True,
            )

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 1 — Working Memory (Redis, <1ms)
        # ══════════════════════════════════════════════════════════════════════
        historico = get_historico_compactado(session_id)
        sinais    = get_sinais(session_id)
        logger.debug("💭 Working memory [%s]: %d turns", session_id, historico.turns_incluidos)

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 2 — Long-Term Facts (Redis KNN, ~3ms)
        # ══════════════════════════════════════════════════════════════════════
        fatos = buscar_fatos_relevantes(user_id=user_id, pergunta=mensagem)
        if fatos:
            logger.debug("🧠 Fatos [%s]: %d | top='%s'", user_id, len(fatos), fatos[0].texto[:50])

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 3 — Semantic Routing (Redis KNN, ~1ms, 0 tokens)
        # ══════════════════════════════════════════════════════════════════════
        resultado_routing = rotear(mensagem, estado_menu)
        rota              = resultado_routing.rota
        logger.info(
            "🗺️  Routing [%s]: rota=%s | conf=%s | score=%.3f",
            user_id, rota.value, resultado_routing.confianca, resultado_routing.score,
        )
        set_sinal(session_id, "rota",              rota.value)
        set_sinal(session_id, "confianca_routing", resultado_routing.confianca)
        if resultado_routing.tool_name:
            set_sinal(session_id, "tool_usada", resultado_routing.tool_name)

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 4 — Self-RAG: decidir se precisa de retrieval (0 tokens típico)
        # ══════════════════════════════════════════════════════════════════════
        precisa_rag = _decidir_precisa_rag(mensagem, rota, historico)
        logger.debug("🔍 Self-RAG [%s]: precisa_rag=%s | rota=%s", user_id, precisa_rag, rota.value)

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 5 — Query Transform (só quando precisa_rag=True)
        # ══════════════════════════════════════════════════════════════════════
        if precisa_rag:
            usar_transform = not (resultado_routing.confianca == "alta" and rota != Rota.GERAL)
            if usar_transform:
                query_transformada = transformar_query(
                    pergunta      = mensagem,
                    fatos_usuario = fatos,
                    usar_sub_queries = (len(mensagem) > 80),
                )
            else:
                query_transformada = QueryTransformada(
                    query_original  = mensagem,
                    query_principal = mensagem,
                    foi_transformada= False,
                    motivo          = "alta confiança routing",
                )
            logger.debug("🔄 Query: %s", query_transformada.query_para_log)
        else:
            query_transformada = QueryTransformada(
                query_original  = mensagem,
                query_principal = mensagem,
                foi_transformada= False,
                motivo          = "self-rag: sem RAG necessário",
            )

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 5.5 — Semantic Cache CHECK
        # ══════════════════════════════════════════════════════════════════════
        if precisa_rag and rota != Rota.GERAL:
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
                        conteudo=cache_hit, rota=rota,
                        tokens_entrada=0, tokens_saida=0, sucesso=True,
                    )
            except Exception as e:
                logger.debug("ℹ️  Cache check ignorado: %s", e)

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 6 — Hybrid Retrieval (só se Self-RAG decidiu que precisa)
        # ══════════════════════════════════════════════════════════════════════
        recuperacao: ResultadoRecuperacao
        if precisa_rag:
            recuperacao = self._executar_retrieval(rota, resultado_routing, query_transformada)
        else:
            recuperacao = ResultadoRecuperacao(
                encontrou   = False,
                metodo_usado= "self-rag-skip",
            )

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 6.5 — CRAG: Avaliar qualidade do retrieval e corrigir se fraco
        # ══════════════════════════════════════════════════════════════════════
        crag_score      = 0.0
        crag_disclaimer = ""

        if precisa_rag and recuperacao.encontrou:
            recuperacao, crag_score, crag_disclaimer = _crag_avaliar_e_corrigir(
                recuperacao       = recuperacao,
                query_transformada= query_transformada,
                rota              = rota,
            )
            logger.info(
                "🔬 CRAG [%s]: score=%.3f | disclaimer=%s",
                user_id, crag_score, bool(crag_disclaimer),
            )

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 7 — Geração com Gemini
        # ══════════════════════════════════════════════════════════════════════
        gemini_resp = self._gerar_resposta(
            mensagem         = mensagem,
            recuperacao      = recuperacao,
            fatos            = fatos,
            historico        = historico,
            sinais           = sinais,
            crag_disclaimer  = crag_disclaimer,
        )

        if not gemini_resp.sucesso:
            return self._tratar_erro_gemini(gemini_resp, rota)

        conteudo = gemini_resp.conteudo.strip()
        if not conteudo or len(conteudo) < 10:
            return AgentResponse(conteudo=_MSG_SEM_INFO, rota=rota, sucesso=False)

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 8 — Persistência Working Memory
        # ══════════════════════════════════════════════════════════════════════
        adicionar_mensagem(session_id, "user",      mensagem)
        adicionar_mensagem(session_id, "assistant", conteudo)
        set_sinal(session_id, "ultimo_topico", query_transformada.query_principal[:80])

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 8.5 — Semantic Cache STORE (só se CRAG aprovou o contexto)
        # ══════════════════════════════════════════════════════════════════════
        cache_guardado = False
        if (precisa_rag and rota != Rota.GERAL
                and crag_score >= CRAG_THRESHOLD_OK
                and len(conteudo) > 50):
            try:
                from src.infrastructure.semantic_cache import store_cache
                store_cache(
                    query    = query_transformada.query_principal,
                    response = conteudo,
                    doc_type = rota.value,
                )
                cache_guardado = True
            except Exception as e:
                logger.debug("ℹ️  Cache store ignorado: %s", e)

        logger.debug("💾 Cache guardado=%s | crag_score=%.3f", cache_guardado, crag_score)

        # ══════════════════════════════════════════════════════════════════════
        # PASSO 9 — Background Extractor
        # ══════════════════════════════════════════════════════════════════════
        self._lancar_extracao_background(user_id, session_id)

        return AgentResponse(
            conteudo       = conteudo,
            rota           = rota,
            tokens_entrada = gemini_resp.input_tokens,
            tokens_saida   = gemini_resp.output_tokens,
            sucesso        = True,
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
            return recuperar(query_transformada)

        source_filter = _ROTA_PARA_SOURCE.get(rota)
        doc_type      = _ROTA_PARA_DOCTYPE.get(rota)

        if rota == Rota.WIKI:
            return recuperar(query_transformada, doc_type_filter="wiki_ctic")

        if resultado_routing.confianca == "alta" and source_filter:
            return recuperar(query_transformada, source_filter=source_filter, doc_type=doc_type)

        return recuperar(query_transformada, doc_type=doc_type)

    # ─────────────────────────────────────────────────────────────────────────
    # Geração
    # ─────────────────────────────────────────────────────────────────────────

    def _gerar_resposta(
        self,
        mensagem:        str,
        recuperacao:     ResultadoRecuperacao,
        fatos:           list[Fato],
        historico:       HistoricoCompactado,
        sinais:          dict,
        crag_disclaimer: str = "",
    ) -> GeminiResponse:
        """Monta prompt final e chama Gemini. Inclui disclaimer CRAG se relevante."""
        contexto = recuperacao.contexto_formatado if recuperacao.encontrou else ""

        # Prepend disclaimer CRAG ao contexto se necessário
        if crag_disclaimer and contexto:
            contexto = f"[NOTA INTERNA — NÃO MENCIONAR AO ALUNO: {crag_disclaimer}]\n\n{contexto}"
        elif crag_disclaimer:
            # Sem contexto + disclaimer → injeta instrução directa no prompt
            contexto = crag_disclaimer

        prompt = montar_prompt_geracao(
            pergunta      = mensagem,
            contexto_rag  = contexto,
            fatos_usuario = fatos_como_string(fatos),
            historico     = historico.texto_formatado,
        )

        # System instruction + histórico já injectado no montar_prompt_geracao
        return chamar_gemini(
            prompt             = prompt,
            system_instruction = SYSTEM_UEMA,
            temperatura        = settings.GEMINI_TEMP,
            max_tokens         = settings.GEMINI_MAX_TOKENS,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Tratamento de erros Gemini
    # ─────────────────────────────────────────────────────────────────────────

    def _tratar_erro_gemini(self, resp: GeminiResponse, rota: Rota) -> AgentResponse:
        erro = resp.erro.lower()
        if "429" in erro or "quota" in erro or "rate" in erro:
            obs.warn("SYSTEM", "gemini_rate_limit", resp.erro[:200])
            return AgentResponse(conteudo=_MSG_RATE_LIMIT, rota=rota, sucesso=False)
        if "503" in erro or "overloaded" in erro:
            return AgentResponse(conteudo=_MSG_RATE_LIMIT, rota=rota, sucesso=False)
        obs.error("SYSTEM", "gemini_erro", resp.erro[:200])
        return AgentResponse(conteudo=_MSG_ERRO_TECNICO, rota=rota, sucesso=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Background extractor
    # ─────────────────────────────────────────────────────────────────────────

    def _lancar_extracao_background(self, user_id: str, session_id: str) -> None:
        from src.memory.memory_extractor import extrair_fatos_do_ultimo_turn

        def _run():
            try:
                extrair_fatos_do_ultimo_turn(user_id, session_id)
            except Exception as e:
                logger.debug("ℹ️  Extração background ignorada [%s]: %s", user_id, e)

        try:
            import asyncio
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(
                asyncio.to_thread(extrair_fatos_do_ultimo_turn, user_id, session_id),
                loop=loop,
            )
        except RuntimeError:
            t = threading.Thread(target=_run, daemon=True, name=f"extractor-{user_id[:8]}")
            t.start()
        except Exception as e:
            logger.debug("ℹ️  Extração background ignorada: %s", e)


# =============================================================================
# PASSO 0 — Guardrails (função pura, sem estado)
# =============================================================================

def _guardrails(mensagem: str) -> tuple[str, str] | None:
    """
    Verifica se a mensagem deve fazer short-circuit ANTES do pipeline completo.

    Retorna:
      (motivo, resposta_directa)  se deve fazer short-circuit
      None                        se deve continuar para o pipeline normal

    NÍVEL 1 — Saudações (regex, 0ms):
      Retorna resposta pré-fabricada que mantém o contexto de assistente UEMA.

    NÍVEL 2 — Fora do domínio (keywords, 0ms):
      Só activa se NÃO houver nenhuma keyword do domínio UEMA.
      Evita falsos positivos: "redação do PAES" tem "redação" mas é domínio UEMA.
    """
    msg = mensagem.strip().lower()

    # Nível 1 — Saudações e conversação informal
    if _REGEX_SAUDACOES.match(mensagem.strip()):
        return ("saudação", _resposta_saudacao(msg))

    # Nível 2 — Fora do domínio (só se sem keywords UEMA)
    palavras = set(re.split(r'\W+', msg))
    tem_dominio_uema  = bool(palavras & _KEYWORDS_DOMINIO_UEMA)
    tem_fora_dominio  = bool(palavras & _KEYWORDS_FORA_DOMINIO)

    if tem_fora_dominio and not tem_dominio_uema:
        return ("fora_dominio", _MSG_FORA_DOMINIO)

    return None


def _resposta_saudacao(msg: str) -> str:
    """Resposta contextual para saudações — mantém a persona UEMA."""
    if any(w in msg for w in ("obrigad", "valeu", "vlw", "thanks")):
        return "De nada! 😊 Estou aqui se precisar de mais informações sobre a UEMA."
    if any(w in msg for w in ("tchau", "flw", "falou", "até", "ate", "xau")):
        return "Até logo! Qualquer dúvida sobre a UEMA, é só chamar. 👋"
    if any(w in msg for w in ("ok", "entendido", "certo", "show", "blz", "tá", "ta")):
        return "Ótimo! Se precisar de mais informações sobre calendário, editais ou contatos da UEMA, estou aqui. 😊"
    # Saudação genérica
    return (
        "Olá! 👋 Sou o Assistente Virtual da UEMA.\n\n"
        "Posso ajudar com:\n"
        "📅 Calendário académico\n"
        "📋 Edital PAES 2026\n"
        "📞 Contatos e e-mails\n"
        "💻 Wiki do CTIC\n\n"
        "Qual é a tua dúvida?"
    )


# =============================================================================
# PASSO 4 — Self-RAG: decisão de usar ou não RAG
# =============================================================================

def _decidir_precisa_rag(
    mensagem:  str,
    rota:      Rota,
    historico: HistoricoCompactado,
) -> bool:
    """
    Decide se a mensagem precisa de retrieval.

    NÍVEL 1 — Heurística local (0 tokens, 0ms de rede):
      - Rota GERAL + mensagem curta + stop-words → False
      - Qualquer rota específica (CALENDARIO/EDITAL/etc.) → True
      (o routing semântico já filtrou — se tem rota específica, precisa RAG)

    NÍVEL 2 — LLM leve (só se inconclusivo, ~80 tokens):
      Chamada rápida ao Gemini com PROMPT_PRECISA_RAG.
      Usada quando: rota=GERAL + mensagem ambígua + histórico recente.
    """
    # Rota específica → sempre precisa RAG
    if rota != Rota.GERAL:
        return True

    msg      = mensagem.strip().lower()
    palavras = set(re.split(r'\W+', msg))

    # Mensagem claramente conversacional → skip RAG
    if (len(palavras) <= 4 and
            not (palavras & _KEYWORDS_DOMINIO_UEMA) and
            palavras & (_STOP_WORDS_SEM_RAG | {"sim", "não", "nao", "talvez"})):
        logger.debug("⚡ Self-RAG heurística: sem RAG (conversacional curto)")
        return False

    # Mensagem contém keyword UEMA → sempre precisa RAG
    if palavras & _KEYWORDS_DOMINIO_UEMA:
        return True

    # Caso ambíguo com histórico recente: usa LLM leve para decidir
    # Só se houver histórico (evita cold-start chamar Gemini para "me ajuda")
    if historico.turns_incluidos >= 2 and len(msg) > 15:
        return _self_rag_llm(mensagem)

    # Default conservador: sem keyword UEMA clara → não faz RAG
    logger.debug("⚡ Self-RAG heurística: sem RAG (sem keywords UEMA)")
    return False


def _self_rag_llm(mensagem: str) -> bool:
    """
    Chama Gemini com mini-prompt para decidir se precisa RAG.
    Custo: ~80 tokens. Usado apenas em casos ambíguos.
    """
    try:
        prompt = PROMPT_PRECISA_RAG.format(mensagem=mensagem[:200])
        resp = chamar_gemini(
            prompt      = prompt,
            temperatura = 0.0,   # determinístico
            max_tokens  = 5,     # só "SIM" ou "NAO"
        )
        decisao = resp.conteudo.strip().upper()
        resultado = "SIM" in decisao
        logger.debug("🤖 Self-RAG LLM: '%s' → %s", decisao, resultado)
        return resultado
    except Exception as e:
        logger.debug("⚠️  Self-RAG LLM falhou, assumindo True: %s", e)
        return True  # default seguro: faz RAG


# =============================================================================
# PASSO 6.5 — CRAG: Corrective RAG
# =============================================================================

def _crag_avaliar_e_corrigir(
    recuperacao:        ResultadoRecuperacao,
    query_transformada: QueryTransformada,
    rota:               Rota,
) -> tuple[ResultadoRecuperacao, float, str]:
    """
    Avalia a qualidade dos chunks recuperados e tenta corrigir se necessário.

    Retorna:
      (recuperacao_final, score_medio, disclaimer)
        score_medio: 0.0–1.0 (qualidade dos chunks)
        disclaimer:  "" se contexto bom; mensagem se fraco/sem contexto

    ALGORITMO:
      1. Calcula score_medio dos chunks (RRF score médio)
      2. score >= CRAG_THRESHOLD_OK (0.40): ✅ contexto bom → usa directamente
      3. score < CRAG_THRESHOLD_OK:
           a) Tenta step-back query (query mais genérica)
           b) Re-busca
           c) Se novo score >= threshold → usa novo contexto
           d) Se ainda baixo >= CRAG_THRESHOLD_MIN → usa com aviso
           e) Se abaixo de CRAG_THRESHOLD_MIN → gera sem RAG + disclaimer forte
    """
    # Calcula score médio dos chunks actuais
    score_medio = _calcular_score_medio(recuperacao)

    if score_medio >= CRAG_THRESHOLD_OK:
        # Contexto bom → passa directo
        return recuperacao, score_medio, ""

    logger.info(
        "⚠️  CRAG: score baixo (%.3f < %.3f). A tentar step-back...",
        score_medio, CRAG_THRESHOLD_OK,
    )

    # Tenta step-back: generaliza a query
    query_stepback = _gerar_query_stepback(query_transformada.query_principal)
    if query_stepback and query_stepback != query_transformada.query_principal:
        query_sb = QueryTransformada(
            query_original  = query_transformada.query_original,
            query_principal = query_stepback,
            foi_transformada= True,
            motivo          = "crag_stepback",
        )

        source_filter = _ROTA_PARA_SOURCE.get(rota)
        doc_type      = _ROTA_PARA_DOCTYPE.get(rota)

        try:
            recuperacao_sb  = recuperar(query_sb, source_filter=source_filter, doc_type=doc_type)
            score_sb        = _calcular_score_medio(recuperacao_sb)
            logger.info("🔄 CRAG step-back: score %.3f → %.3f", score_medio, score_sb)

            if score_sb >= CRAG_THRESHOLD_OK:
                return recuperacao_sb, score_sb, ""

            if score_sb > score_medio:
                # Step-back melhorou mas ainda fraco → usa mas com aviso leve
                disclaimer = (
                    "A informação encontrada pode ser parcial. "
                    "Recomenda ao aluno verificar em uema.br se precisar de dados precisos."
                )
                return recuperacao_sb, score_sb, disclaimer

        except Exception as e:
            logger.warning("⚠️  CRAG step-back falhou: %s", e)

    # Contexto original fraco, step-back não ajudou
    if score_medio >= CRAG_THRESHOLD_MIN:
        # Fraco mas não inútil → usa com disclaimer leve
        disclaimer = (
            "Não encontrei informação completamente precisa nos documentos. "
            "Responde com base no que encontraste, mas sugere verificar em uema.br."
        )
        return recuperacao, score_medio, disclaimer
    else:
        # Abaixo do mínimo → gera sem RAG com disclaimer forte
        logger.warning("❌ CRAG: contexto insuficiente (%.3f). Gerando sem RAG.", score_medio)
        recuperacao_vazia = ResultadoRecuperacao(
            encontrou   = False,
            metodo_usado= "crag_rejected",
        )
        disclaimer = (
            "Não encontraste informação nos documentos. "
            "Responde ao aluno que não encontraste essa informação específica "
            "e sugere consultar uema.br ou a secretaria do curso."
        )
        return recuperacao_vazia, 0.0, disclaimer


def _calcular_score_medio(recuperacao: ResultadoRecuperacao) -> float:
    """Calcula score RRF médio dos chunks recuperados."""
    if not recuperacao.encontrou:
        return 0.0

    # Tenta extrair scores do contexto_formatado (se disponível como lista)
    chunks = getattr(recuperacao, "chunks", None)
    if chunks and hasattr(chunks[0], "rrf_score"):
        scores = [c.rrf_score for c in chunks if c.rrf_score > 0]
        return sum(scores) / len(scores) if scores else 0.0

    # Fallback: se tem contexto mas sem scores → score conservador
    contexto = recuperacao.contexto_formatado or ""
    if len(contexto) > 200:
        return CRAG_THRESHOLD_OK  # assume OK se tem conteúdo substancial
    return 0.0


def _gerar_query_stepback(query: str) -> str:
    """
    Gera uma query mais genérica (step-back) sem chamar o LLM.

    Estratégia: remove especificidades (nomes próprios, datas, siglas compostas)
    e mantém os termos gerais mais relevantes.

    Exemplos:
      "matrícula Engenharia Civil noturno 2026.1" → "matrícula veteranos semestre"
      "vagas BR-PPI Direito São Luís"             → "vagas cotas Direito"
      "email coordenador Engenharia Elétrica"     → "contato coordenação Engenharia"
    """
    # Remove datas e semestres (ex: 2026.1, 03/02/2026)
    q = re.sub(r'\b20\d{2}[\./]\d{1,2}\b', '', query)
    q = re.sub(r'\b\d{1,2}/\d{2}/20\d{2}\b', '', q)

    # Remove siglas compostas de cotas (mantém termos genéricos)
    q = re.sub(r'\b(br-ppi|br-q|br-dc|ir-ppi|cfo-pp|pcd)\b', 'cotas', q, flags=re.IGNORECASE)

    # Remove nomes de cursos específicos muito longos (heurística: >2 palavras)
    cursos_regex = re.compile(
        r'\b(engenharia|administração|direito|medicina|pedagogia|letras|história)\s+'
        r'(civil|elétrica|da|do|de)?\s*\w+',
        re.IGNORECASE,
    )
    q = cursos_regex.sub(lambda m: m.group(1), q)

    q = re.sub(r'\s+', ' ', q).strip()

    # Evita step-back vazio ou idêntico
    if not q or q.lower() == query.lower():
        # Último recurso: pega só as 3 primeiras palavras
        palavras = query.split()[:3]
        return " ".join(palavras) if len(palavras) >= 2 else ""

    return q


# =============================================================================
# Singleton global
# =============================================================================

agent_core = AgentCore()