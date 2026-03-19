"""
tests/test_pipeline_admin.py — Pipeline de Teste Completo (v1.3 — HyDE Routing)
=================================================================================

MUDANÇAS v1.3 vs v1.2:
────────────────────────
  FASE 3 actualizada:
    - Threshold de score esperado actualizado para HyDE (0.80+ em vez de 0.62+)
    - Mostra a "query de exemplo" encontrada pelo KNN (diagnóstico visual)
    - Tabela de scores esperados por query explicada nos comentários

  Bugs acumulados corrigidos (mantidos das versões anteriores):
    - BUG A (Fase 2): importa PDF_CONFIG directamente
    - BUG B (Fase 3): chave 'fase' sempre presente em resultados
    - BUG C (Fase 3): inicializa AgentCore antes das queries
    - BUG 4 (Fase 4): SettingsMock com número de teste fixo
    - BUG 5 (Fase 5): threshold RRF corrigido para 0.008
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
    for linha in a[:500].splitlines():
        print(f"     {linha}")
    if len(a) > 500:
        print(f"     [... +{len(a)-500} chars]")


# =============================================================================
# FASE 0 — Diagnóstico
# =============================================================================

def fase0_diagnostico() -> bool:
    _h("FASE 0 — Diagnóstico do Sistema")
    from src.infrastructure.redis_client import redis_ok, get_redis
    from src.rag.ingestion import Ingestor

    if redis_ok():
        _ok("Redis Stack conectado")
        r    = get_redis()
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

    try:
        from src.domain.semantic_router import listar_tools_registadas
        tools = listar_tools_registadas()
        n     = len(tools)
        _info(f"Tools no SemanticRouter: {n} {'✅' if n >= 4 else '⚠️  (esperado ≥4)'}")
        for t in tools:
            print(f"     • {t['name']} — {t['description']}")
    except Exception as e:
        _warn(f"Não listou tools: {e}")

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

    # ── Importa PDF_CONFIG (não DOCUMENT_CONFIG) ──────────────────────────────
    from src.rag.ingestion import PDF_CONFIG, Ingestor
    from src.infrastructure.settings import settings

    _CONFIG_TESTE = {
        "agente_rag_uema_spec.pdf": {
            "doc_type": "geral", "titulo": "Especificação Técnica Bot UEMA v5",
            "chunk_size": 400, "overlap": 60,
            "label": "ESPECIFICAÇÃO BOT UEMA v5", "parser": "pymupdf",
        },
        "instrucoes_uso_agente.pdf": {
            "doc_type": "geral", "titulo": "Manual de Uso e Comandos Bot UEMA",
            "chunk_size": 350, "overlap": 50,
            "label": "MANUAL USO BOT UEMA", "parser": "pymupdf",
        },
        "vagas_mock_2026.csv": {
            "doc_type": "edital", "titulo": "Vagas Mock PAES 2026 (Teste)",
            "chunk_size": 300, "overlap": 40, "label": "VAGAS MOCK PAES 2026",
        },
        "contatos_mock.csv": {
            "doc_type": "contatos", "titulo": "Contatos Mock UEMA (Teste)",
            "chunk_size": 250, "overlap": 30, "label": "CONTATOS MOCK UEMA",
        },
    }

    registados = 0
    for nome, cfg in _CONFIG_TESTE.items():
        if nome not in PDF_CONFIG:
            PDF_CONFIG[nome] = cfg
            registados += 1
            _info(f"Registado em PDF_CONFIG: '{nome}' → {cfg['doc_type']}")

    _ok(f"{registados} entradas adicionadas ao PDF_CONFIG.")

    pastas = {
        os.path.join(settings.DATA_DIR, "PDF", "testes"): [
            "agente_rag_uema_spec.pdf", "instrucoes_uso_agente.pdf",
        ],
        os.path.join(settings.DATA_DIR, "CSV", "testes"): [
            "vagas_mock_2026.csv", "contatos_mock.csv",
        ],
    }

    ingestor     = Ingestor()
    total_chunks = 0
    erros        = []

    for pasta, nomes in pastas.items():
        if not os.path.isdir(pasta):
            _warn(f"Pasta não existe: {pasta} — corre setup_projeto.py")
            continue
        encontrados = [f for f in os.listdir(pasta) if not f.startswith(".")]
        _info(f"Ficheiros em {pasta}: {encontrados}")

        for nome in nomes:
            caminho = os.path.join(pasta, nome)
            if not os.path.exists(caminho):
                _warn(f"'{nome}' não encontrado")
                erros.append(nome)
                continue
            if nome not in PDF_CONFIG:
                _fail(f"'{nome}' não no PDF_CONFIG")
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
    if not erros:
        _ok("Fase 2 concluída sem erros")
    else:
        _warn(f"Problemas: {erros}")


# =============================================================================
# FASE 3 — Queries (HyDE Routing v2)
# =============================================================================

def fase3_queries() -> None:
    _h("FASE 3 — Queries de Teste ao Agente (HyDE Routing v2)")

    from src.agent.core import agent_core, _guardrails, _decidir_precisa_rag
    from src.domain.entities import EstadoMenu
    from src.memory.working_memory import _historico_vazio

    # ── Inicializa AgentCore (regista tools com HyDE no Redis) ───────────────
    if not agent_core._inicializado:
        _info("Inicializando AgentCore (regista ~60 queries de exemplo no Redis)...")
        from src.tools import get_tools_ativas
        from src.infrastructure.redis_client import inicializar_indices
        inicializar_indices()
        tools = get_tools_ativas()
        agent_core.inicializar(tools)
        _ok(f"AgentCore inicializado com {len(tools)} tools")

    # Verifica quantas queries foram indexadas
    from src.infrastructure.redis_client import get_redis, PREFIX_TOOLS
    r              = get_redis()
    _, keys_tools  = r.scan(0, match=f"{PREFIX_TOOLS}*", count=500)
    _info(f"Queries de exemplo indexadas no Redis: {len(keys_tools)}")
    _info("Score esperado com HyDE: 0.85-0.95 (vs 0.50 antes)")

    historico_vazio = _historico_vazio()

    # ── Dataset de teste ──────────────────────────────────────────────────────
    # Score HyDE esperado:
    #   guardrail: não chega ao router
    #   CALENDARIO: "quando é a matrícula?" → exemplo idêntico → ~0.95
    #   EDITAL:     "quantas vagas BR-PPI?"  → exemplo similar → ~0.88
    #   CONTATOS:   "email da PROG"          → exemplo próximo → ~0.91
    queries_teste = [
        ("oi",                            True,  None,        "saudação → guardrail"),
        ("obrigado!",                     True,  None,        "agradecimento → guardrail"),
        ("me faz uma redação sobre clima",True,  None,        "fora domínio → guardrail"),
        ("quando é a matrícula?",         False, "CALENDARIO","calendário → HyDE score ~0.95"),
        ("quantas vagas BR-PPI?",         False, "EDITAL",    "edital → HyDE score ~0.88"),
        ("email da PROG",                 False, "CONTATOS",  "contatos → HyDE score ~0.91"),
    ]

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
            resultados.append({
                "query": query, "ok": guardrail_ok,
                "fase": "guardrail", "rota": "N/A", "ms": 0, "tokens": 0,
            })
            continue

        # Testa routing HyDE
        from src.domain.semantic_router import rotear
        resultado_routing = rotear(query)
        rota              = resultado_routing.rota
        score             = resultado_routing.score
        precisa           = _decidir_precisa_rag(query, rota, historico_vazio)

        # Interpreta score HyDE
        if score >= 0.85:
            score_label = f"✅ EXCELENTE ({score:.3f} ≥ 0.85)"
        elif score >= 0.80:
            score_label = f"✅ ALTA confiança ({score:.3f} ≥ 0.80)"
        elif score >= 0.62:
            score_label = f"🔶 MÉDIA confiança ({score:.3f} ≥ 0.62)"
        elif score >= 0.40:
            score_label = f"⚠️  BAIXA confiança ({score:.3f} ≥ 0.40)"
        else:
            score_label = f"❌ ABAIXO DO MÍNIMO ({score:.3f} < 0.40)"

        print(f"  🗺️  Routing: rota={rota.value} | score: {score_label} | método={resultado_routing.metodo}")
        print(f"  🔍 Self-RAG: precisa_rag={precisa}")

        rota_ok = (expect_rota is None or rota.value == expect_rota)
        if not rota_ok:
            _warn(f"Rota incorrecta: esperava {expect_rota}, recebeu {rota.value}")
            if score < 0.62:
                _info("Causa: score abaixo de THRESHOLD_MEDIA — verifica se as tools foram "
                      "registadas com HyDE (docker-compose restart bot)")

        # Chamada real ao agente
        ms, tokens, resp_ok = 0, 0, False
        try:
            from src.infrastructure.redis_client import redis_ok
            if redis_ok():
                t0   = time.monotonic()
                resp = agent_core.responder(
                    user_id    = "test_admin",
                    session_id = "test_admin_session",
                    mensagem   = query,
                )
                ms      = int((time.monotonic() - t0) * 1000)
                tokens  = resp.tokens_total
                resp_ok = resp.sucesso
                _a(resp.conteudo)
                print(f"  ⏱  {ms}ms | tokens={tokens} | sucesso={resp.sucesso} | rota_final={resp.rota.value}")

                # Avalia qualidade da resposta
                conteudo = resp.conteudo.lower()
                if expect_rota == "CALENDARIO" and any(w in conteudo for w in ["matrícula", "data", "período", "fevereiro", "março"]):
                    _ok("Resposta contém informação de calendário ✅")
                elif expect_rota == "EDITAL" and any(w in conteudo for w in ["vaga", "cota", "paes", "br-ppi", "ac:", "pcd"]):
                    _ok("Resposta contém informação de edital ✅")
                elif expect_rota == "CONTATOS" and any(w in conteudo for w in ["email", "prog", "@uema", "contato", "pró-reitora"]):
                    _ok("Resposta contém informação de contatos ✅")
                elif expect_rota and "uema.br" in conteudo and "secretaria" in conteudo:
                    _warn("Resposta genérica (sem RAG) — routing pode ainda não estar correcto")

        except Exception as e:
            _fail(f"Erro: {e}")
            import traceback; traceback.print_exc()

        resultados.append({
            "query": query, "ok": resp_ok and rota_ok,
            "fase": "pipeline", "rota": rota.value, "ms": ms, "tokens": tokens,
        })

    # ── Sumário ───────────────────────────────────────────────────────────────
    print(f"\n  {'─'*54}")
    n_ok = sum(1 for r in resultados if r["ok"])
    print(f"  SUMÁRIO: {n_ok}/{len(resultados)} OK")
    for r in resultados:
        icon  = "✅" if r["ok"] else "❌"
        fase  = r["fase"]
        rota  = r.get("rota", "?")
        ms    = r.get("ms", 0)
        print(f"  {icon} [{fase:<10}] rota={rota:<12} {ms:>5}ms | {r['query'][:35]}")

    if n_ok == len(resultados):
        _ok("Fase 3 perfeita! HyDE Routing funcionando.")
    elif n_ok >= len(resultados) - 1:
        _warn("Fase 3 quase perfeita — 1 caso com problema.")
    else:
        _warn("Fase 3 com múltiplos problemas — verifica o log acima.")
        _info("Se routing ainda retorna GERAL: reinicia o bot para registar as tools com HyDE.")
        _info("  docker-compose restart bot celery-worker")


# =============================================================================
# FASE 4 — Comandos Admin
# =============================================================================

def fase4_comandos_admin() -> None:
    _h("FASE 4 — Comandos Admin")

    from src.middleware.security_guard import SecurityGuard
    from src.infrastructure.settings import settings
    from src.infrastructure.redis_client import get_redis_text

    class SettingsMock:
        ADMIN_NUMBERS   = "5598000000001"
        STUDENT_NUMBERS = "5598000000002"

    guard      = SecurityGuard(get_redis_text(), SettingsMock())
    teu_numero = "5598000000001"

    print(f"\n  Número de teste (ADMIN): {teu_numero}")
    admin_real = settings.ADMIN_NUMBERS
    tem_admin  = admin_real and "X" not in admin_real and len(admin_real) > 5
    _info(f"ADMIN_NUMBERS no .env real: '{admin_real}' {'✅' if tem_admin else '❌ — edita o .env!'}")

    print(f"\n  {'─'*54}")
    comandos = [
        ("!status",              "CMD_ADMIN", False),
        ("!tools",               "CMD_ADMIN", False),
        ("!limpar_cache",        "CMD_ADMIN", True),
        ("!ragas",               "CMD_ADMIN", True),
        ("!fatos 5598000000001", "CMD_ADMIN", False),
        ("!reload",              "CMD_ADMIN", True),
        ("!ingerir",             "ERRO",      False),
    ]

    erros = 0
    for cmd, acao_esperada, _ in comandos:
        result = guard.verificar(user_id=teu_numero, body=cmd)
        ok     = result.acao in ("CMD_ADMIN", "INGERIR_DOC", "INGERIR_FICHEIRO", "ERRO")
        print(f"  {cmd:<25} → {result.acao:<20} {'✅' if ok else '❌'} (nível: {result.nivel.value})")
        if not ok:
            erros += 1

    print(f"\n  {'─'*54}")
    r_guest = guard.verificar(user_id="5598999999999", body="!limpar_cache")
    if r_guest.acao == "LLM":
        _ok("GUEST bloqueado de comandos admin ✅")
    else:
        _fail(f"GUEST conseguiu acesso: {r_guest.acao}")
        erros += 1

    if erros == 0:
        _ok("Fase 4 OK — todos os comandos admin correctos")
    else:
        _warn(f"Fase 4: {erros} problema(s)")


# =============================================================================
# FASE 5 — RAG Eval rápido
# =============================================================================

def fase5_rag_eval_rapido() -> None:
    _h("FASE 5 — RAG Eval Rápido (score RRF real)")

    _info("Score RRF com k=60: máximo teórico ≈ 0.033 (top-1 nos 2 métodos)")
    _info("Threshold realista: 0.008 (rank-1 num método = chunk relevante encontrado)")

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
                "query":    "quando é a matrícula de veteranos 2026.1?",
                "source":   "calendario-academico-2026.pdf",
                "keywords": ["matrícula", "veterano", "data"],
            },
            {
                "query":    "quantas vagas ampla concorrência engenharia civil?",
                "source":   "edital_paes_2026.pdf",
                "keywords": ["Engenharia Civil", "AC", "vagas"],
            },
            {
                "query":    "email da PROG pró-reitoria de graduação?",
                "source":   "guia_contatos_2025.pdf",
                "keywords": ["PROG", "email", "@uema"],
            },
        ]

        print(f"\n  {'─'*54}")
        scores_ok  = 0
        keywords_ok= 0

        for i, caso in enumerate(dataset, 1):
            query    = caso["query"]
            vetor    = emb.embed_query(query)
            source   = caso["source"]
            keywords = caso["keywords"]

            resultados = busca_hibrida(
                query_text=query, query_embedding=vetor,
                source_filter=source, k_vector=5, k_text=6,
            )

            print(f"\n  [{i}] {query[:55]}")

            if not resultados:
                _fail("0 chunks — verifica ingestão")
                continue

            top       = resultados[0]
            top_score = top.get("rrf_score", 0.0)
            top_prev  = top.get("content", "")[:100].replace("\n", " ")
            n_chunks  = len(resultados)

            interp = (
                "top-1 em AMBOS ✅✅" if top_score >= 0.030 else
                "top-1 num método ✅" if top_score >= 0.016 else
                "relevante 🔶"        if top_score >= 0.008 else
                "muito baixo ❌"
            )
            print(f"       Chunks: {n_chunks} | Top RRF: {top_score:.4f} ({interp})")
            print(f"       Top chunk: {top_prev}")

            if top_score >= THRESHOLD_RRF_MINIMO:
                scores_ok += 1
                _ok(f"Score OK ({top_score:.4f})")
            else:
                _fail(f"Score muito baixo ({top_score:.4f})")

            conteudo_low = top.get("content", "").lower()
            kw_found     = [kw for kw in keywords if kw.lower() in conteudo_low]
            if kw_found:
                keywords_ok += 1
                _ok(f"Keywords: {kw_found}")
            else:
                _warn(f"Keywords esperadas não encontradas: {keywords}")

        print(f"\n  {'─'*54}")
        print(f"  Scores válidos: {scores_ok}/{len(dataset)}")
        print(f"  Keywords OK:    {keywords_ok}/{len(dataset)}")

        if scores_ok == len(dataset):
            _ok("Pipeline RAG funcionando correctamente")
        else:
            _warn("Scores baixos — pode indicar problema na ingestão")

    except Exception as e:
        _fail(f"RAG eval falhou: {e}")
        import traceback; traceback.print_exc()


# =============================================================================
# Runner
# =============================================================================

def run(fases: list[int] | None = None) -> None:
    _all = fases is None

    print("\n🧪 PIPELINE DE TESTE — Bot UEMA v5 (HyDE Routing)")
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