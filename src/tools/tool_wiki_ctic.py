"""
tools/tool_wiki_ctic.py — Tool de Consulta à Wiki do CTIC/UEMA (v1.0)
=======================================================================

O QUE FAZ:
───────────
  Scrapa a Wiki DokuWiki do CTIC (https://ctic.uema.br/wiki/) e indexa
  o conteúdo no Redis Stack para busca híbrida (BM25 + Vetor).

  FLUXO:
    1. scrape_wiki_page(url) → extrai conteúdo da div.page + links internos
    2. Converte HTML → Markdown (preserva # ## para chunking hierárquico)
    3. Cache Redis com TTL 24h → não sobrecarrega o servidor da UEMA
    4. indexar_wiki(urls) → scrapa + embeda + salva chunks no Redis Stack
    5. get_tool_wiki_ctic() → LangChain tool para o AgentCore

DESIGN — DIV.PAGE:
───────────────────
  O DokuWiki renderiza todo o conteúdo útil dentro de <div class="page">.
  Ao focar apenas nesta div ignoramos:
    - sidebar (~15% do HTML)
    - header com links de navegação (~10%)
    - links de "Ações da Página" (edit, rev, backlink, recent)
  Economia: ~40% menos tokens de ruído na ingestão.

DESIGN — CACHE REDIS:
──────────────────────
  Chave: wiki:cache:{url_hash}
  TTL:   24h (86400s)
  Valor: {"url": "...", "content": "markdown...", "links": [...], "ts": ...}

  Durante desenvolvimento, o servidor da UEMA não é sobrecarregado:
  a primeira chamada faz o HTTP; as seguintes usam o cache Redis.
  Para forçar re-scraping: redis-cli DEL wiki:cache:*

DESIGN — MARKDOWNIFY:
──────────────────────
  HTML → Markdown preserva a hierarquia de títulos (H1→#, H2→##).
  Para RAG isto é crucial: o modelo sabe que um parágrafo está
  "abaixo de # Serviços de TI" sem precisar de embedding de todo o HTML.
  Resultado: chunks mais contextuais, menos alucinações em respostas sobre a Wiki.

INTEGRAÇÃO:
────────────
  1. Adicionar ao tools/__init__.py:
       from src.tools.tool_wiki_ctic import get_tool_wiki_ctic
       # em get_tools_ativas(): get_tool_wiki_ctic()

  2. Adicionar à entities.py:
       WIKI = "WIKI"  # na enum Rota

  3. Adicionar ao semantic_router.py:
       _TOOL_PARA_ROTA: {"consultar_wiki_ctic": Rota.WIKI, ...}

  4. Adicionar ao core.py:
       _ROTA_PARA_SOURCE: {Rota.WIKI: "wiki_ctic", ...}

  5. Indexar a Wiki (uma vez, ou periodicamente):
       from src.tools.tool_wiki_ctic import indexar_wiki
       indexar_wiki()  # scrapa todas as páginas conhecidas

DEPENDÊNCIAS (adicionar ao requirements.txt):
──────────────────────────────────────────────
  httpx>=0.27       # cliente HTTP assíncrono/síncrono
  beautifulsoup4>=4.12
  markdownify>=0.12
  lxml              # parser HTML mais rápido para BeautifulSoup
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import TypedDict

import httpx
from bs4 import BeautifulSoup

from src.infrastructure.redis_client import get_redis_text, salvar_chunk
from src.rag.ingestion import _criar_chunks, _gerar_chunk_id, _limpar_texto

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

WIKI_BASE_URL  = "https://ctic.uema.br/wiki/doku.php"
WIKI_DOC_TYPE  = "wiki_ctic"
WIKI_SOURCE    = "wiki_ctic"

CACHE_PREFIX   = "wiki:cache:"
CACHE_TTL      = 86400          # 24h — refresca diariamente

CHUNK_CONFIG   = {
    "doc_type":   WIKI_DOC_TYPE,
    "titulo":     "Wiki CTIC/UEMA",
    "chunk_size": 400,
    "overlap":    60,
    "label":      "WIKI CTIC/UEMA",
}

# Páginas raiz a indexar (o scraper descobre links internos automaticamente)
WIKI_SEED_PAGES = [
    f"{WIKI_BASE_URL}?id=start",
    f"{WIKI_BASE_URL}?id=servicos",
    f"{WIKI_BASE_URL}?id=suporte",
    f"{WIKI_BASE_URL}?id=sistemas",
    f"{WIKI_BASE_URL}?id=redes",
    f"{WIKI_BASE_URL}?id=infraestrutura",
]

# User-Agent realista — evita bloqueios simples de bot
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Links de acções administrativas do DokuWiki a ignorar
_ACOES_ADMIN = frozenset({"do=edit", "do=revisions", "do=backlink",
                           "do=recent", "do=index", "do=login",
                           "do=register", "do=resendpwd", "do=admin"})


# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

class WikiPageResult(TypedDict):
    content: str          # Markdown do conteúdo da div.page
    links:   list[str]    # URLs internas descobertas
    url:     str          # URL original
    cached:  bool         # True se veio do cache Redis


# ─────────────────────────────────────────────────────────────────────────────
# Scraper core
# ─────────────────────────────────────────────────────────────────────────────

def scrape_wiki_page(url: str, usar_cache: bool = True) -> WikiPageResult:
    """
    Extrai conteúdo de uma página DokuWiki e retorna Markdown + links internos.

    Parâmetros:
      url:        URL completa com parâmetro ?id= (ex: ...?id=servicos)
      usar_cache: Se True, verifica cache Redis antes de fazer HTTP request

    Retorna:
      WikiPageResult com keys: content (Markdown), links (lista URLs), url, cached

    ALGORITMO:
      1. Verifica cache Redis → retorna imediatamente se hit
      2. Faz GET com httpx + User-Agent real
      3. Parseia com BeautifulSoup
      4. Extrai div.page (conteúdo útil)
      5. Remove elementos de administração (sidebar, actions, toolbar)
      6. Extrai links internos (?id=) e normaliza URLs
      7. Converte HTML → Markdown com markdownify
      8. Limpa Markdown (espaços, linhas duplicadas)
      9. Guarda no cache Redis
     10. Retorna resultado
    """
    # ── Cache Redis ───────────────────────────────────────────────────────────
    if usar_cache:
        cached = _get_cache(url)
        if cached:
            logger.debug("🗃️  Cache hit: %s", url)
            return cached

    logger.info("🌐 Scraping: %s", url)

    # ── HTTP Request ──────────────────────────────────────────────────────────
    try:
        with httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            timeout=15.0,
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except httpx.HTTPStatusError as e:
        logger.warning("⚠️  HTTP %d para %s", e.response.status_code, url)
        return WikiPageResult(content="", links=[], url=url, cached=False)
    except Exception as e:
        logger.error("❌ Falha ao aceder %s: %s", url, e)
        return WikiPageResult(content="", links=[], url=url, cached=False)

    # ── Parse HTML ────────────────────────────────────────────────────────────
    soup = BeautifulSoup(html, "lxml")

    # ── Filtro: extrai apenas div.page (padrão DokuWiki) ─────────────────────
    div_page = soup.find("div", class_="page")
    if not div_page:
        # Fallback: tenta div#dokuwiki__content
        div_page = soup.find("div", id="dokuwiki__content")
    if not div_page:
        logger.warning("⚠️  div.page não encontrada em %s. Usando body.", url)
        div_page = soup.find("body") or soup

    # ── Remove elementos de ruído ─────────────────────────────────────────────
    for ruido in div_page.find_all(["div", "section"], class_=[
        "toolbar", "secedit", "footnotes", "catlist",
        "plugin_tag", "docInfo", "breadcrumbs",
    ]):
        ruido.decompose()

    # ── Extrai links internos ─────────────────────────────────────────────────
    links = _extrair_links_internos(div_page, url)

    # ── Converte HTML → Markdown ──────────────────────────────────────────────
    markdown = _html_to_markdown(div_page)
    markdown = _limpar_markdown(markdown)

    if not markdown.strip():
        logger.warning("⚠️  Markdown vazio para %s", url)
        return WikiPageResult(content="", links=links, url=url, cached=False)

    resultado = WikiPageResult(
        content=markdown,
        links=links,
        url=url,
        cached=False,
    )

    # ── Guarda no cache Redis ─────────────────────────────────────────────────
    _set_cache(url, resultado)

    logger.info(
        "✅ Wiki scrapeada: %d chars Markdown | %d links | %s",
        len(markdown), len(links), url,
    )
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# Indexação no Redis Stack
# ─────────────────────────────────────────────────────────────────────────────

def indexar_pagina_wiki(url: str, forcar: bool = False) -> int:
    """
    Scrapa uma página e indexa os chunks no Redis Stack.

    Parâmetros:
      url:    URL da página Wiki
      forcar: Se True, ignora cache e re-scrapa

    Retorna:
      Número de chunks indexados (0 se falha ou vazio).
    """
    from src.rag.embeddings import get_embeddings

    resultado = scrape_wiki_page(url, usar_cache=not forcar)
    if not resultado["content"]:
        return 0

    page_id   = _url_para_page_id(url)
    source_id = f"wiki:{page_id}"

    # Cria config específico para esta página
    config = {
        **CHUNK_CONFIG,
        "titulo": f"Wiki CTIC — {page_id}",
        "label":  f"WIKI CTIC | {page_id.upper()}",
    }

    texto_limpo = _limpar_texto(resultado["content"])
    chunks      = list(_criar_chunks(texto_limpo, source_id, config))
    if not chunks:
        return 0

    embeddings_model = get_embeddings()
    embeddings       = embeddings_model.embed_documents([c.texto_puro for c in chunks])

    for chunk, emb in zip(chunks, embeddings):
        salvar_chunk(
            chunk_id    = _gerar_chunk_id(source_id, chunk.chunk_index),
            content     = chunk.texto_final,
            source      = source_id,
            doc_type    = WIKI_DOC_TYPE,
            embedding   = emb,
            chunk_index = chunk.chunk_index,
            metadata    = {
                **chunk.metadata,
                "wiki_url":     url,
                "wiki_page_id": page_id,
            },
        )

    logger.info("📚 Wiki indexada: '%s' → %d chunks.", source_id, len(chunks))
    return len(chunks)


def indexar_wiki(
    seed_urls: list[str] | None = None,
    max_paginas: int = 50,
    forcar: bool = False,
) -> dict[str, int]:
    """
    Indexação completa da Wiki: scrapa as seed URLs e segue links internos.

    Parâmetros:
      seed_urls:   URLs de partida (default: WIKI_SEED_PAGES)
      max_paginas: Limite de páginas para evitar indexação infinita
      forcar:      Se True, ignora cache e re-scrapa tudo

    Retorna:
      dict {url: n_chunks_indexados}
    """
    if seed_urls is None:
        seed_urls = WIKI_SEED_PAGES

    visitadas: set[str] = set()
    fila:      list[str] = list(seed_urls)
    resultado: dict[str, int] = {}

    while fila and len(visitadas) < max_paginas:
        url = fila.pop(0)
        if url in visitadas:
            continue

        visitadas.add(url)
        n_chunks = indexar_pagina_wiki(url, forcar=forcar)
        resultado[url] = n_chunks

        if n_chunks > 0:
            # Descobre novos links e adiciona à fila
            page_result = scrape_wiki_page(url, usar_cache=True)
            for link in page_result["links"]:
                if link not in visitadas and link not in fila:
                    fila.append(link)

        # Pausa educada — não sobrecarrega o servidor
        time.sleep(0.5)

    total = sum(resultado.values())
    logger.info(
        "✅ Indexação Wiki concluída: %d páginas | %d chunks totais",
        len(visitadas), total,
    )
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# LangChain Tool
# ─────────────────────────────────────────────────────────────────────────────

def get_tool_wiki_ctic():
    """
    Fábrica: configura e retorna a @tool para consulta à Wiki do CTIC.
    """
    from langchain_core.tools import tool
    from src.infrastructure.redis_client import busca_hibrida
    from src.rag.embeddings import get_embeddings

    embeddings_model = get_embeddings()

    @tool
    def consultar_wiki_ctic(query: str) -> str:
        """
        Consulta a Wiki do CTIC (Centro de Tecnologia da Informação e Comunicação da UEMA).

        Use para perguntas sobre:
          - Sistemas de TI da UEMA (SIGAA, SIE, e-mail institucional)
          - Suporte técnico de informática e redes
          - Serviços do CTIC (laboratórios, infraestrutura, VPN)
          - Manuais e tutoriais de sistemas institucionais
          - Senhas, acessos, credenciais institucionais
          - Wi-Fi e conectividade nos campi

        Parâmetro query: o que o utilizador quer saber sobre TI da UEMA.
        """
        import unicodedata

        def norm(t: str) -> str:
            return unicodedata.normalize("NFD", t).encode("ascii", "ignore").decode().lower()

        try:
            query_norm = norm(query)
            vetor      = embeddings_model.embed_query(query_norm)

            resultados = busca_hibrida(
                query_text     = query_norm,
                query_embedding= vetor,
                source_filter  = None,    # Não filtra por source — busca em todas as páginas Wiki
                k_vector       = 5,
                k_text         = 6,
            )

            # Filtra apenas chunks da Wiki
            chunks_wiki = [
                r for r in resultados
                if str(r.get("doc_type", "")).startswith("wiki")
                or str(r.get("source",   "")).startswith("wiki")
            ]

            if not chunks_wiki:
                return (
                    "Não encontrei essa informação na Wiki do CTIC. "
                    "Tente com outras palavras como: SIGAA, suporte, senha, "
                    "laboratório, rede, sistema, e-mail institucional."
                )

            blocos   = [r["content"].strip() for r in chunks_wiki if r.get("content", "").strip()]
            resposta = "\n---\n".join(blocos)
            if len(resposta) > 1500:
                resposta = resposta[:1500] + "\n[...resultado truncado]"
            return resposta

        except Exception as e:
            logger.exception("❌ Erro na tool Wiki CTIC: %s", e)
            return "ERRO TÉCNICO NA FERRAMENTA — não tente novamente nesta resposta."

    return consultar_wiki_ctic


# ─────────────────────────────────────────────────────────────────────────────
# Cache Redis
# ─────────────────────────────────────────────────────────────────────────────

def _get_cache(url: str) -> WikiPageResult | None:
    r_txt = get_redis_text()
    try:
        key  = _cache_key(url)
        data = r_txt.get(key)
        if data:
            doc = json.loads(data)
            return WikiPageResult(**doc)
    except Exception:
        pass
    return None


def _set_cache(url: str, resultado: WikiPageResult) -> None:
    r_txt = get_redis_text()
    try:
        key  = _cache_key(url)
        data = json.dumps({**resultado, "ts": int(time.time())})
        r_txt.setex(key, CACHE_TTL, data)
    except Exception as e:
        logger.debug("⚠️  Falha ao guardar cache wiki: %s", e)


def _cache_key(url: str) -> str:
    return f"{CACHE_PREFIX}{hashlib.md5(url.encode()).hexdigest()[:16]}"


def limpar_cache_wiki() -> int:
    """Remove todo o cache da Wiki (útil para forçar re-scraping)."""
    r_txt = get_redis_text()
    deletados = 0
    cursor = 0
    while True:
        cursor, keys = r_txt.scan(cursor, match=f"{CACHE_PREFIX}*", count=500)
        if keys:
            r_txt.delete(*keys)
            deletados += len(keys)
        if cursor == 0:
            break
    logger.info("🗑️  Cache wiki limpo: %d entradas.", deletados)
    return deletados


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários internos
# ─────────────────────────────────────────────────────────────────────────────

def _extrair_links_internos(div_page, url_base: str) -> list[str]:
    """
    Extrai todos os links internos da Wiki (href contendo ?id=).
    Ignora links de acções administrativas (edit, rev, etc.).
    """
    links_encontrados: list[str] = []
    vistos: set[str] = set()

    for a_tag in div_page.find_all("a", href=True):
        href = a_tag["href"]

        # Filtra: deve conter "id=" (padrão DokuWiki)
        if "id=" not in href:
            continue

        # Filtra: ignora acções administrativas
        if any(acao in href for acao in _ACOES_ADMIN):
            continue

        # Normaliza URL
        if href.startswith("http"):
            url_completa = href
        elif href.startswith("/"):
            # URL relativa ao domínio
            from urllib.parse import urlparse
            parsed = urlparse(url_base)
            url_completa = f"{parsed.scheme}://{parsed.netloc}{href}"
        else:
            # URL relativa à página actual
            url_completa = f"{WIKI_BASE_URL}?{href.lstrip('?')}" if "?" in href else f"{WIKI_BASE_URL}?{href}"

        # Remove fragmentos e parâmetros desnecessários
        url_limpa = _normalizar_url_wiki(url_completa)

        if url_limpa and url_limpa not in vistos:
            vistos.add(url_limpa)
            links_encontrados.append(url_limpa)

    return links_encontrados


def _normalizar_url_wiki(url: str) -> str:
    """Normaliza URL da Wiki: mantém apenas domínio + ?id=valor."""
    try:
        from urllib.parse import urlparse, parse_qs, urlencode
        parsed = urlparse(url)

        # Só aceita URLs do domínio da UEMA
        if "ctic.uema.br" not in parsed.netloc and "uema.br" not in parsed.netloc:
            return ""

        params = parse_qs(parsed.query)
        if "id" not in params:
            return ""

        page_id = params["id"][0]
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?id={page_id}"
    except Exception:
        return ""


def _url_para_page_id(url: str) -> str:
    """Extrai o page_id de uma URL Wiki (?id=page_id)."""
    try:
        from urllib.parse import urlparse, parse_qs
        params = parse_qs(urlparse(url).query)
        return params.get("id", ["start"])[0]
    except Exception:
        return "unknown"


def _html_to_markdown(elemento) -> str:
    """
    Converte elemento BeautifulSoup para Markdown usando markdownify.
    Fallback para get_text() se markdownify não estiver instalado.
    """
    html_str = str(elemento)
    try:
        from markdownify import markdownify as md
        return md(
            html_str,
            heading_style="ATX",      # # H1, ## H2 (não underline)
            bullets="-",              # listas com -
            strip=["img", "figure"],  # remove imagens (inúteis para RAG)
        )
    except ImportError:
        logger.warning("⚠️  markdownify não instalado. Usando texto simples. "
                       "pip install markdownify")
        from bs4 import BeautifulSoup as BS
        return BS(html_str, "html.parser").get_text(separator="\n", strip=True)


def _limpar_markdown(texto: str) -> str:
    """Remove artefactos de conversão HTML→Markdown."""
    texto = re.sub(r"\n{3,}", "\n\n", texto)            # máx 2 linhas em branco
    texto = re.sub(r"[ \t]+\n", "\n", texto)            # espaços no fim de linha
    texto = re.sub(r"\[edit\]|\[rev\]|\[top\]", "", texto)  # links de acção residuais
    texto = re.sub(r"\*\*\s*\*\*", "", texto)           # bold vazio
    texto = re.sub(r"#{1,6}\s*\n", "", texto)           # headers vazios
    return texto.strip()