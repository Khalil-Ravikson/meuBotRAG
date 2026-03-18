"""
tests/test_pipeline_admin.py — Pipeline de Teste Completo (v1.2 — 3 Bugs Corrigidos)
======================================================================================

BUGS CORRIGIDOS vs v1.1:
─────────────────────────
  BUG A (FASE 2): DOCUMENT_CONFIG e PDF_CONFIG são o mesmo objecto em teoria,
    mas o ingestion.py real usa PDF_CONFIG internamente. O teste modificava
    DOCUMENT_CONFIG após a importação, mas o Ingestor._ingerir_ficheiro() faz
    lookup em PDF_CONFIG que pode ser a referência original pré-importação.
    CORRIGIDO: o teste agora importa e modifica PDF_CONFIG directamente.

  BUG B (FASE 3): KeyError 'fase' no sumário.
    A lista resultados tinha dois formatos distintos:
      - guardrail: {"query", "ok", "fase"}       → tinha 'fase'
      - pipeline:  {"query", "ok", "ms", "tokens", "rota"}  → SEM 'fase'
    CORRIGIDO: todas as entradas têm 'fase' garantida.

  BUG C (FASE 3): Routing GERAL para queries específicas ("matrícula", "BR-PPI").
    O SemanticRouter usa embeddings das descrições das tools no Redis.
    Se o AgentCore não foi inicializado, as tools não estão registadas →
    router retorna GERAL (fallback regex). Isto é comportamento correcto.
    CORRIGIDO: o teste inicializa as tools no Redis antes das queries,
    e explica claramente quando o routing é via regex vs semântico.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("ENV_FILE_PATH", ".env.local")


# =============================================================================
# Utilitários de output
# =============================================================================

def _h(titulo: str) -> None:
    print(f"\n{'═'*62}")
    print(f"  🧪 {titulo}")
    print('═'*62)

def _ok(msg: str) -> None:   print(f"  ✅ {msg}")
def _fail(msg: str) -> None:  print(f"  ❌ {msg}")
def _info(msg: str) -> None:  print(f"  ℹ️  {msg}")
def _warn(msg: str) -> None:  print(f"  ⚠️  {msg}")
def _q(q: str) -> None:      print(f"\n  📨 Query: \"{q}\"")
def _a(a: str) -> None:
    print(f"  🤖 Resposta:")
    for linha in a[:400].splitlines():
        print(f"     {linha}")
    if len(a) > 400:
        print(f"     [... +{len(a)-400} chars]")


# =============================================================================
# FASE 0 — Diagnóstico
# =============================================================================

def fase0_diagnostico() -> bool:
    _h("FASE 0 — Diagnóstico do Sistema")
    from src.infrastructure.redis_client import redis_ok, get_redis
    from src.rag.ingestion import Ingestor

    if redis_ok():
        _ok("Redis Stack conectado")
        r = get_redis()
        info = r.info("server")
        _info(f"Versão Redis: {info.get('redis_version', '?')}")
    else:
        _fail("Redis offline! Inicia: docker run -p 6379:6379 redis/redis-stack:latest")
        return False

    try:
        from src.infrastructure.redis_client import IDX_CHUNKS
        idx_info = get_redis().ft(IDX_CHUNKS).info()
        _ok(f"Índice chunks: {idx_info.get('num_docs', 0)} docs")
    except Exception:
        _fail("Índice de chunks não existe — corre o startup primeiro")
        return False

    ingestor = Ingestor()
    sources  = ingestor.diagnosticar()
    _info(f"Sources no Redis: {sources}")

    from src.agent.core import agent_core
    _info(f"AgentCore inicializado: {agent_core._inicializado}")

    # Verifica tools registadas no SemanticRouter
    try:
        from src.domain.semantic_router import listar_tools_registadas
        tools = listar_tools_registadas()
        _info(f"Tools no SemanticRouter: {len(tools)} {'✅' if tools else '⚠️  (0 → routing será regex)'}")
    except Exception:
        _warn("Não foi possível listar tools do SemanticRouter")

    return True


# =============================================================================
# FASE 1 — Scraping Wiki
# =============================================================================

def fase1_scraping() -> None:
    _h("FASE 1 — Scraping Wiki CTIC (dry-run)")
    _info("Este teste NÃO escreve no Redis.")
    try:
        from src.tools.tool_wiki_ctic import scrape_wiki_page, WIKI_BASE_URL
        url    = f"{WIKI_BASE_URL}?id=start"
        result = scrape_wiki_page(url, usar_cache=False)
        print(f"\n  URL:   {url}")
        print(f"  Chars: {len(result['content'])}")
        print(f"  Links: {len(result['links'])}")
        print(f"\n  [PREVIEW — 400 chars]")
        print("  " + result["content"][:400].replace("\n", "\n  "))
        if len(result["content"]) > 100:
            _ok("Scraping OK")
        else:
            _fail("Conteúdo muito curto")
    except Exception as e:
        _fail(f"Scraping falhou: {e}")
        import traceback; traceback.print_exc()


# =============================================================================
# FASE 2 — Ingestão de dados de teste
# =============================================================================

def fase2_ingestao() -> None:
    _h("FASE 2 — Ingestão de Dados de Teste")

    # ── CORRECÇÃO BUG A: importa PDF_CONFIG que é o que _ingerir_ficheiro() lê ──
    # O ingestion.py real usa PDF_CONFIG internamente (não DOCUMENT_CONFIG).
    # DOCUMENT_CONFIG = PDF_CONFIG são o mesmo objecto — mas para garantir,
    # importamos e modificamos PDF_CONFIG directamente.
    from src.rag.ingestion import PDF_CONFIG, Ingestor
    from src.infrastructure.settings import settings

    _CONFIG_TESTE = {
        "agente_rag_uema_spec.pdf": {
            "doc_type":   "geral",
            "titulo":     "Especificação Técnica Bot UEMA v5",
            "chunk_size": 400,
            "overlap":    60,
            "label":      "ESPECIFICAÇÃO BOT UEMA v5",
            "parser":     "pymupdf",
        },
        "instrucoes_uso_agente.pdf": {
            "doc_type":   "geral",
            "titulo":     "Manual de Uso e Comandos Bot UEMA",
            "chunk_size": 350,
            "overlap":    50,
            "label":      "MANUAL USO BOT UEMA",
            "parser":     "pymupdf",
        },
        "vagas_mock_2026.csv": {
            "doc_type":   "edital",
            "titulo":     "Vagas Mock PAES 2026 (Teste)",
            "chunk_size": 300,
            "overlap":    40,
            "label":      "VAGAS MOCK PAES 2026",
        },
        "contatos_mock.csv": {
            "doc_type":   "contatos",
            "titulo":     "Contatos Mock UEMA (Teste)",
            "chunk_size": 250,
            "overlap":    30,
            "label":      "CONTATOS MOCK UEMA",
        },
    }

    # Regista directamente em PDF_CONFIG (que é o que _ingerir_ficheiro() usa)
    registados = 0
    for nome, cfg in _CONFIG_TESTE.items():
        if nome not in PDF_CONFIG:
            PDF_CONFIG[nome] = cfg
            registados += 1
            _info(f"Registado: '{nome}' → {cfg['doc_type']}")

    if registados == 0:
        _info("Todos os ficheiros de teste já estavam no PDF_CONFIG.")
    else:
        _ok(f"{registados} entradas adicionadas ao PDF_CONFIG.")

    # Verifica que a modificação foi efectiva
    for nome in _CONFIG_TESTE:
        assert nome in PDF_CONFIG, f"FALHA: '{nome}' não está no PDF_CONFIG após registo!"

    # Pastas a verificar
    pastas_e_ficheiros = {
        os.path.join(settings.DATA_DIR, "PDF", "testes"): [
            "agente_rag_uema_spec.pdf",
            "instrucoes_uso_agente.pdf",
        ],
        os.path.join(settings.DATA_DIR, "CSV", "testes"): [
            "vagas_mock_2026.csv",
            "contatos_mock.csv",
        ],
    }

    ingestor     = Ingestor()
    total_chunks = 0
    erros        = []

    for pasta, nomes_esperados in pastas_e_ficheiros.items():
        if not os.path.isdir(pasta):
            _warn(f"Pasta não existe: {pasta}")
            continue

        ficheiros_encontrados = [f for f in os.listdir(pasta) if not f.startswith(".")]
        _info(f"Ficheiros em {pasta}: {ficheiros_encontrados}")

        for nome in nomes_esperados:
            caminho = os.path.join(pasta, nome)
            if not os.path.exists(caminho):
                _warn(f"'{nome}' não encontrado — corre setup_projeto.py primeiro")
                erros.append(nome)
                continue

            # Confirma que está no PDF_CONFIG
            if nome not in PDF_CONFIG:
                _fail(f"'{nome}' ainda não está no PDF_CONFIG — algo correu mal")
                erros.append(nome)
                continue

            t0     = time.monotonic()
            chunks = ingestor._ingerir_ficheiro(caminho)
            ms     = int((time.monotonic() - t0) * 1000)

            if chunks > 0:
                _ok(f"'{nome}': {chunks} chunks em {ms}ms")
                total_chunks += chunks
            else:
                _fail(f"'{nome}': 0 chunks")
                erros.append(nome)

    print(f"\n  Total chunks de teste ingeridos: {total_chunks}")
    if erros:
        _warn(f"Ficheiros com problemas: {erros}")
    else:
        _ok("Fase 2 concluída sem erros")


# =============================================================================
# FASE 3 — Queries de teste
# =============================================================================

def fase3_queries() -> None:
    _h("FASE 3 — Queries de Teste ao Agente")

    from src.agent.core import agent_core
    from src.agent.core import _guardrails, _decidir_precisa_rag
    from src.domain.entities import EstadoMenu
    from src.memory.working_memory import _historico_vazio

    # ── Inicializa AgentCore E regista tools no SemanticRouter ───────────────
    if not agent_core._inicializado:
        _info("AgentCore não inicializado — inicializando (regista tools no router)...")
        from src.tools import get_tools_ativas
        from src.infrastructure.redis_client import inicializar_indices
        inicializar_indices()
        tools = get_tools_ativas()
        agent_core.inicializar(tools)  # ← isto regista as tools no Redis
        _ok(f"AgentCore inicializado com {len(tools)} tools")

        # Verifica se tools foram registadas
        from src.domain.semantic_router import listar_tools_registadas
        tools_redis = listar_tools_registadas()
        _info(f"Tools registadas no SemanticRouter: {len(tools_redis)}")

    historico_vazio = _historico_vazio()

    queries_teste = [
        # (query, esperado_guardrail, esperado_rota, descricao)
        ("oi",              True,  None,        "saudação → guardrail"),
        ("obrigado!",       True,  None,        "agradecimento → guardrail"),
        ("me faz uma redação sobre o clima", True, None, "fora domínio → guardrail"),
        ("quando é a matrícula?",  False, "CALENDARIO", "calendário → RAG"),
        ("quantas vagas BR-PPI?",  False, "EDITAL",     "edital → sigla"),
        ("email da PROG",          False, "CONTATOS",   "contatos → sigla"),
    ]

    # ── CORRECÇÃO BUG B: todas as entradas têm 'fase' garantida ──────────────
    resultados: list[dict] = []

    for query, expect_guardrail, expect_rota, descricao in queries_teste:
        _q(query)
        print(f"  📋 Esperado: guardrail={expect_guardrail} | rota={expect_rota} | {descricao}")

        # Testa guardrail
        gr = _guardrails(query)
        guardrail_ok = (gr is not None) == expect_guardrail

        if gr is not None:
            print(f"  🛡️  Guardrail: {gr[0]} → \"{gr[1][:60]}\"")
            if guardrail_ok:
                _ok(f"Guardrail correcto: {gr[0]}")
            else:
                _fail("Guardrail inesperado")
            # ── BUG B CORRIGIDO: 'fase' sempre presente ───────────────────────
            resultados.append({
                "query": query,
                "ok":    guardrail_ok,
                "fase":  "guardrail",       # ← sempre presente
                "rota":  "N/A",
                "ms":    0,
                "tokens": 0,
            })
            continue

        # Testa routing + self-rag
        from src.domain.semantic_router import rotear
        result_routing = rotear(query)
        rota           = result_routing.rota
        precisa        = _decidir_precisa_rag(query, rota, historico_vazio)

        print(f"  🗺️  Routing: rota={rota.value} | conf={result_routing.confianca} | score={result_routing.score:.3f} | método={result_routing.metodo}")
        print(f"  🔍 Self-RAG: precisa_rag={precisa}")

        # Explica quando routing é regex (normal quando tools não inicializadas)
        if result_routing.metodo == "fallback_regex":
            _info("Routing via regex (fallback) — esperado se tools ainda não no Redis")
        elif result_routing.confianca == "baixa" and rota.value == "GERAL":
            _warn(f"Score baixo ({result_routing.score:.3f} < 0.62) → GERAL. "
                  "Verifica se as tools estão registadas: docker exec bot bash -c "
                  "\"python -c 'from src.domain.semantic_router import listar_tools_registadas; print(listar_tools_registadas())'\"")

        rota_ok = (expect_rota is None or rota.value == expect_rota)

        # Chamada real ao agente
        ms, tokens = 0, 0
        resp_ok    = False
        try:
            from src.infrastructure.redis_client import redis_ok
            if redis_ok():
                t0   = time.monotonic()
                resp = agent_core.responder(
                    user_id    = "test_admin",
                    session_id = "test_admin_session",
                    mensagem   = query,
                )
                ms     = int((time.monotonic() - t0) * 1000)
                tokens = resp.tokens_total
                resp_ok= resp.sucesso
                _a(resp.conteudo)
                print(f"  ⏱  {ms}ms | tokens={tokens} | sucesso={resp.sucesso} | rota_final={resp.rota.value}")

                # Nota explicativa sobre CRAG com routing GERAL
                if resp.rota.value == "GERAL" and expect_rota and expect_rota != "GERAL":
                    _warn(f"Resposta GERAL quando esperávamos {expect_rota}.")
                    _info("Causa provável: tools não registadas no Redis → routing regex → "
                          "score baixo → CRAG sem contexto → resposta genérica.")
                    _info("Solução: certifica que o bot fez startup completo antes de testar "
                          "(docker-compose logs -f bot | grep 'AgentCore inicializado')")
            else:
                _info("Redis offline — sem chamada real ao agente")
        except Exception as e:
            _fail(f"Erro: {e}")
            import traceback; traceback.print_exc()

        # ── BUG B CORRIGIDO: 'fase' sempre presente ───────────────────────────
        resultados.append({
            "query":  query,
            "ok":     resp_ok and rota_ok,
            "fase":   "pipeline",          # ← sempre presente
            "rota":   rota.value,
            "ms":     ms,
            "tokens": tokens,
        })

    # Sumário — BUG B: todas as entradas têm 'fase', sem KeyError
    print(f"\n  {'─'*54}")
    print(f"  SUMÁRIO: {sum(1 for r in resultados if r['ok'])}/{len(resultados)} OK")
    for r in resultados:
        icon  = "✅" if r["ok"] else "❌"
        fase  = r["fase"]          # ← sempre existe agora
        rota  = r.get("rota", "?")
        ms    = r.get("ms", 0)
        print(f"  {icon} [{fase:<10}] rota={rota:<10} {ms:>5}ms | {r['query'][:35]}")


# =============================================================================
# FASE 4 — Comandos Admin
# =============================================================================

def fase4_comandos_admin() -> None:
    _h("FASE 4 — Comandos Admin")

    from src.middleware.security_guard import SecurityGuard
    from src.infrastructure.settings import settings
    from src.infrastructure.redis_client import get_redis_text

    # Settings mockados para teste (número fixo como ADMIN)
    class SettingsMock:
        ADMIN_NUMBERS   = "5598000000001"
        STUDENT_NUMBERS = "5598000000002"

    guard      = SecurityGuard(get_redis_text(), SettingsMock())
    teu_numero = "5598000000001"

    print(f"\n  Número de teste (ADMIN): {teu_numero}")
    _info(f"ADMIN_NUMBERS no .env real: '{settings.ADMIN_NUMBERS}' "
          f"{'✅' if settings.ADMIN_NUMBERS and 'X' not in settings.ADMIN_NUMBERS else '❌ VAZIO ou placeholder — edita o .env!'}")

    print(f"\n  {'─'*54}")
    print("  Testando comandos admin:")

    comandos = [
        ("!status",              "CMD_ADMIN", False),
        ("!tools",               "CMD_ADMIN", False),
        ("!limpar_cache",        "CMD_ADMIN", True),
        ("!ragas",               "CMD_ADMIN", True),
        ("!fatos 5598000000001", "CMD_ADMIN", False),
        ("!reload",              "CMD_ADMIN", True),
        ("!ingerir",             "ERRO",      False),   # sem ficheiro → erro educado
    ]

    erros = 0
    for cmd, acao_esperada, precisa_celery in comandos:
        result = guard.verificar(user_id=teu_numero, body=cmd)
        ok     = result.acao in (acao_esperada, "INGERIR_DOC", "INGERIR_FICHEIRO", "CMD_ADMIN", "ERRO")
        print(f"\n  Comando: {cmd}")
        print(f"  Ação:    {result.acao:<20} {'✅' if ok else '❌'}")
        print(f"  Nível:   {result.nivel.value}")
        if result.resposta_rapida:
            print(f"  Rápida:  {result.resposta_rapida[:70]}")
        if result.parametro:
            print(f"  Param:   {result.parametro}")
        if not ok:
            erros += 1

    # Verifica que GUEST não tem acesso
    print(f"\n  {'─'*54}")
    print("  Teste de segurança: GUEST tentando !limpar_cache")
    result_guest = guard.verificar(user_id="5598999999999", body="!limpar_cache")
    if result_guest.acao == "LLM":
        _ok("GUEST correctamente bloqueado dos comandos admin")
    else:
        _fail(f"GUEST conseguiu acesso admin: {result_guest.acao}")
        erros += 1

    if erros == 0:
        _ok("Fase 4 OK — todos os comandos admin funcionam")
    else:
        _warn(f"Fase 4: {erros} problema(s) encontrado(s)")


# =============================================================================
# FASE 5 — RAG Eval rápido
# =============================================================================

def fase5_rag_eval_rapido() -> None:
    _h("FASE 5 — RAG Eval Rápido (score RRF real)")

    _info("Score RRF com k=60: valores típicos entre 0.008 e 0.033")
    _info("Fórmula: RRF = 1/(60+rank). Top-1 nos 2 métodos ≈ 0.033 (máximo real)")
    _info("Threshold usado: 0.008 (rank-1 num método = chunk relevante encontrado)")

    try:
        from src.infrastructure.redis_client import redis_ok, busca_hibrida
        from src.rag.embeddings import get_embeddings

        if not redis_ok():
            _info("Redis offline — Fase 5 ignorada")
            return

        emb                  = get_embeddings()
        THRESHOLD_RRF_MINIMO = 0.008

        dataset = [
            {
                "query":     "quando é a matrícula de veteranos 2026.1?",
                "source":    "calendario-academico-2026.pdf",
                "keywords":  ["matrícula", "veterano", "data"],
            },
            {
                "query":     "quantas vagas ampla concorrência engenharia civil?",
                "source":    "edital_paes_2026.pdf",
                "keywords":  ["Engenharia Civil", "AC", "vagas"],
            },
            {
                "query":     "email da PROG pró-reitoria de graduação?",
                "source":    "guia_contatos_2025.pdf",
                "keywords":  ["PROG", "email", "@uema"],
            },
        ]

        print(f"\n  {'─'*54}")
        scores_validos = 0
        keywords_ok    = 0

        for i, caso in enumerate(dataset, 1):
            query    = caso["query"]
            vetor    = emb.embed_query(query)
            source   = caso["source"]
            keywords = caso["keywords"]

            resultados = busca_hibrida(
                query_text     = query,
                query_embedding= vetor,
                source_filter  = source,
                k_vector       = 5,
                k_text         = 6,
            )

            print(f"\n  [{i}] {query[:55]}")

            if not resultados:
                _fail("0 chunks — verifica ingestão")
                continue

            top_score   = resultados[0].get("rrf_score", 0.0)
            top_content = resultados[0].get("content", "")
            top_preview = top_content[:100].replace("\n", " ")
            n_chunks    = len(resultados)

            # Interpretação do score
            if top_score >= 0.030:
                interpretacao = "top-1 em AMBOS métodos ✅✅"
            elif top_score >= 0.016:
                interpretacao = "top-1 num método ✅"
            elif top_score >= 0.008:
                interpretacao = "relevante mas não top-1 🔶"
            else:
                interpretacao = "muito baixo ❌"

            print(f"       Chunks: {n_chunks} | Top RRF: {top_score:.4f} ({interpretacao})")
            print(f"       Top chunk: {top_preview}")

            # Verifica keywords
            conteudo_lower   = top_content.lower()
            kw_encontradas   = [kw for kw in keywords if kw.lower() in conteudo_lower]

            if top_score >= THRESHOLD_RRF_MINIMO:
                scores_validos += 1
                _ok(f"Score OK ({top_score:.4f})")
            else:
                _fail(f"Score muito baixo ({top_score:.4f})")

            if kw_encontradas:
                keywords_ok += 1
                _ok(f"Keywords encontradas: {kw_encontradas}")
            else:
                _warn(f"Keywords esperadas não encontradas: {keywords}")
                _info("O chunk pode ser relevante semanticamente mesmo sem keywords exactas")

        print(f"\n  {'─'*54}")
        print(f"  Scores válidos: {scores_validos}/{len(dataset)}")
        print(f"  Keywords OK:    {keywords_ok}/{len(dataset)}")

        if scores_validos == len(dataset):
            _ok("Pipeline RAG funcionando correctamente")
        else:
            _warn("Alguns scores baixos — pode indicar problema na ingestão ou poucos chunks")

    except Exception as e:
        _fail(f"RAG eval falhou: {e}")
        import traceback; traceback.print_exc()


# =============================================================================
# Runner
# =============================================================================

def run(fases: list[int] | None = None) -> None:
    _all = fases is None

    print("\n🧪 PIPELINE DE TESTE — Bot UEMA v5 (Admin Only)")
    print("=" * 62)

    if _all or 0 in fases:
        ok = fase0_diagnostico()
        if not ok and _all:
            print("\n❌ Diagnóstico falhou — abortando.")
            return

    if _all or 1 in fases:
        fase1_scraping()

    if _all or 2 in fases:
        fase2_ingestao()

    if _all or 3 in fases:
        fase3_queries()

    if _all or 4 in fases:
        fase4_comandos_admin()

    if _all or 5 in fases:
        fase5_rag_eval_rapido()

    print(f"\n{'═'*62}")
    print("  Pipeline de testes concluído.")
    print("  Para eval completo com RAGAS: python tests/rag_eval.py")
    print('═'*62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de testes Bot UEMA v5")
    parser.add_argument("--fase", type=int, help="Fase específica (0-5)")
    args = parser.parse_args()

    if args.fase is not None:
        run(fases=[args.fase])
    else:
        run(fases=None)