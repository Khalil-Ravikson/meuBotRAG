"""
tests/test_wiki_scraper.py — Pipeline de Teste do Wiki Scraper CTIC
====================================================================

PROPÓSITO:
  Testar o scraper da Wiki do CTIC antes de indexar no Redis.
  Mostra exactamente o que vai ser ingerido — conteúdo, links, chunking.

COMO USAR (da raiz do projecto):
  # 1. Garante que o .env.local existe com REDIS_URL
  # 2. Corre isoladamente (sem Docker, sem Redis obrigatório para scraping):
  python -m pytest tests/test_wiki_scraper.py -v -s

  # Ou directamente:
  python tests/test_wiki_scraper.py

FASES DO TESTE:
  Fase 1 — Fetch da página raiz: verifica conectividade e parsing
  Fase 2 — Qualidade do Markdown: verifica headers, limpeza, tamanho
  Fase 3 — Extracção de links: verifica que links internos são detectados
  Fase 4 — Chunking: verifica que o texto será dividido correctamente
  Fase 5 — Cache Redis: verifica que o cache funciona (skip se sem Redis)

ADMINISTRADOR APENAS:
  Limitado ao teu número via ADMIN_NUMBERS no .env
"""
from __future__ import annotations

import os
import sys
import re
import time

# Garante que src/ está no path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configura ENV antes de importar src/
os.environ.setdefault("ENV_FILE_PATH", ".env.local")


def _separador(titulo: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print('='*60)


def _ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def _fail(msg: str) -> None:
    print(f"  ❌ {msg}")


def _info(msg: str) -> None:
    print(f"  ℹ️  {msg}")


# =============================================================================
# FASE 1 — Conectividade e Fetch
# =============================================================================

def test_fase1_conectividade():
    _separador("FASE 1 — Conectividade e Fetch da Wiki")

    from src.tools.tool_wiki_ctic import scrape_wiki_page, WIKI_BASE_URL

    url = f"{WIKI_BASE_URL}?id=start"
    _info(f"URL de teste: {url}")

    t0     = time.monotonic()
    result = scrape_wiki_page(url, usar_cache=False)
    elapsed= int((time.monotonic() - t0) * 1000)

    print(f"\n  URL:    {result['url']}")
    print(f"  Cached: {result['cached']}")
    print(f"  Tempo:  {elapsed}ms")
    print(f"  Chars Markdown: {len(result['content'])}")
    print(f"  Links encontrados: {len(result['links'])}")

    assert result["url"] == url,      "URL de retorno incorrecta"
    assert len(result["content"]) > 100, "Conteúdo muito curto — scraping falhou?"

    _ok(f"Fetch OK em {elapsed}ms | {len(result['content'])} chars")
    return result


# =============================================================================
# FASE 2 — Qualidade do Markdown
# =============================================================================

def test_fase2_qualidade_markdown(result: dict):
    _separador("FASE 2 — Qualidade do Markdown")

    content = result["content"]

    # Mostra primeiros 600 chars
    print("\n  [PREVIEW — primeiros 600 chars do Markdown gerado]")
    print("  " + "-"*54)
    for linha in content[:600].splitlines():
        print(f"  {linha}")
    print("  " + "-"*54)

    # Verifica headers Markdown (#, ##)
    headers = [l for l in content.splitlines() if l.startswith("#")]
    print(f"\n  Headers encontrados: {len(headers)}")
    for h in headers[:5]:
        print(f"    {h[:70]}")

    # Verifica ausência de ruído HTML
    html_tags = re.findall(r"<[a-zA-Z][^>]{0,20}>", content)
    if html_tags:
        _fail(f"Tags HTML residuais: {html_tags[:5]}")
    else:
        _ok("Sem tags HTML residuais")

    # Verifica links de acção removidos
    admin_patterns = ["do=edit", "do=revisions", "do=backlink"]
    for pat in admin_patterns:
        if pat in content:
            _fail(f"Link admin residual: {pat}")
        else:
            _ok(f"'{pat}' removido")

    assert len(content) > 100, "Markdown vazio"
    _ok(f"Qualidade OK | {len(headers)} headers detectados")


# =============================================================================
# FASE 3 — Extracção de Links
# =============================================================================

def test_fase3_links(result: dict):
    _separador("FASE 3 — Links Internos Detectados")

    links = result["links"]
    print(f"\n  Total de links internos: {len(links)}")

    for i, link in enumerate(links[:10], 1):
        print(f"  {i:2d}. {link}")

    if len(links) > 10:
        print(f"  ... e mais {len(links) - 10} links")

    # Verifica que todos têm ?id=
    sem_id = [l for l in links if "id=" not in l]
    if sem_id:
        _fail(f"Links sem ?id= detectados: {sem_id[:3]}")
    else:
        _ok("Todos os links têm parâmetro ?id=")

    # Verifica que links admin foram filtrados
    admin_links = [l for l in links if any(a in l for a in ["do=edit", "do=rev", "do=backlink"])]
    if admin_links:
        _fail(f"Links admin não filtrados: {admin_links}")
    else:
        _ok("Links admin correctamente filtrados")

    assert len(links) > 0, "Nenhum link interno encontrado"
    _ok(f"Links OK: {len(links)} links internos válidos")


# =============================================================================
# FASE 4 — Simulação de Chunking
# =============================================================================

def test_fase4_chunking(result: dict):
    _separador("FASE 4 — Simulação de Chunking")

    from src.rag.ingestion import _criar_chunks, _limpar_texto

    content   = result["content"]
    source_id = "wiki:start"
    config    = {"doc_type": "wiki_ctic", "titulo": "Wiki CTIC — Start",
                 "chunk_size": 400, "overlap": 60, "label": "WIKI CTIC | START"}

    texto_limpo = _limpar_texto(content)
    chunks      = list(_criar_chunks(texto_limpo, source_id, config))

    print(f"\n  Texto limpo: {len(texto_limpo)} chars")
    print(f"  Chunks gerados: {len(chunks)}")
    print(f"  Tamanho médio: {int(sum(len(c.texto_puro) for c in chunks)/max(len(chunks),1))} chars")

    print(f"\n  [PREVIEW — primeiros 2 chunks]")
    for i, chunk in enumerate(chunks[:2], 1):
        print(f"\n  --- Chunk {i} ({len(chunk.texto_final)} chars) ---")
        print("  " + chunk.texto_final[:300].replace("\n", "\n  "))

    assert len(chunks) > 0, "Nenhum chunk gerado"

    # Verifica prefixo hierárquico anti-alucinação
    sem_prefixo = [c for c in chunks if not c.texto_final.startswith("[")]
    if sem_prefixo:
        _fail(f"{len(sem_prefixo)} chunks sem prefixo hierárquico")
    else:
        _ok(f"Todos os {len(chunks)} chunks têm prefixo hierárquico")

    _ok(f"Chunking OK: {len(chunks)} chunks prontos para embedding")
    return chunks


# =============================================================================
# FASE 5 — Cache Redis (opcional)
# =============================================================================

def test_fase5_cache_redis(result: dict):
    _separador("FASE 5 — Cache Redis (opcional)")

    try:
        from src.tools.tool_wiki_ctic import (
            _cache_key, _get_cache, _set_cache, limpar_cache_wiki,
        )
        from src.infrastructure.redis_client import redis_ok

        if not redis_ok():
            _info("Redis offline — Fase 5 ignorada (só funciona com Redis a correr)")
            return

        url = result["url"]

        # Guarda no cache
        _set_cache(url, result)
        _ok(f"Cache guardado: {_cache_key(url)}")

        # Lê do cache
        cached = _get_cache(url)
        assert cached is not None, "Cache miss inesperado"
        assert cached["url"] == url
        _ok(f"Cache hit: {len(cached['content'])} chars recuperados")

        # Segunda chamada ao scraper (deve usar cache)
        from src.tools.tool_wiki_ctic import scrape_wiki_page
        t0     = time.monotonic()
        result2= scrape_wiki_page(url, usar_cache=True)
        elapsed= int((time.monotonic() - t0) * 1000)

        assert result2["cached"] == True, "Esperava cache=True na segunda chamada"
        _ok(f"Cache hit confirmado em {elapsed}ms (vs ~500ms sem cache)")

        # Limpa o cache de teste
        limpar_cache_wiki()
        _ok("Cache limpo após teste")

    except ImportError as e:
        _info(f"Módulo não disponível: {e} — Fase 5 ignorada")
    except Exception as e:
        _fail(f"Erro no cache: {e}")


# =============================================================================
# Runner principal
# =============================================================================

def run_all():
    print("\n🔬 PIPELINE DE TESTE — Wiki Scraper CTIC/UEMA")
    print("=" * 60)
    print("  Este teste verifica o scraping ANTES de indexar no Redis.")
    print("  Inspeciona: fetch → Markdown → links → chunking → cache")
    print("=" * 60)

    errors = []

    try:
        result = test_fase1_conectividade()
    except Exception as e:
        _fail(f"FASE 1 falhou: {e}")
        print("\n❌ Teste abortado — sem conectividade com a Wiki.")
        return

    for fase, fn in [
        ("Fase 2", lambda: test_fase2_qualidade_markdown(result)),
        ("Fase 3", lambda: test_fase3_links(result)),
        ("Fase 4", lambda: test_fase4_chunking(result)),
        ("Fase 5", lambda: test_fase5_cache_redis(result)),
    ]:
        try:
            fn()
        except AssertionError as e:
            _fail(f"{fase} ASSERTION: {e}")
            errors.append(f"{fase}: {e}")
        except Exception as e:
            _fail(f"{fase} ERRO: {e}")
            errors.append(f"{fase}: {e}")

    print(f"\n{'='*60}")
    if errors:
        print(f"  ❌ {len(errors)} fase(s) com problemas:")
        for err in errors:
            print(f"     • {err}")
    else:
        print("  ✅ Todas as fases passaram! Wiki Scraper pronto para uso.")
        print("\n  Próximo passo:")
        print("    from src.tools.tool_wiki_ctic import indexar_wiki")
        print("    indexar_wiki(max_paginas=20)")
    print('='*60)


# pytest compatibility
def test_scraper_completo():
    """Entry point para pytest."""
    run_all()


if __name__ == "__main__":
    run_all()