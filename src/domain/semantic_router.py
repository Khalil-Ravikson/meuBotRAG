"""
domain/semantic_router.py — Roteamento Semântico via Redis (v4)
================================================================

MUDANÇAS v4:
  _TOOL_PARA_ROTA: adicionado "consultar_wiki_ctic" → Rota.WIKI
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from functools import lru_cache

from src.domain.entities import EstadoMenu, Rota
from src.infrastructure.redis_client import (
    IDX_TOOLS,
    PREFIX_TOOLS,
    VECTOR_DIM,
    get_redis,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD_ALTA   = 0.80
THRESHOLD_MEDIA  = 0.62
THRESHOLD_MINIMO = 0.40

# ─────────────────────────────────────────────────────────────────────────────
# Mapeamentos
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_PARA_ROTA: dict[str, Rota] = {
    "consultar_calendario_academico": Rota.CALENDARIO,
    "consultar_edital_paes_2026":     Rota.EDITAL,
    "consultar_contatos_uema":        Rota.CONTATOS,
    "consultar_wiki_ctic":            Rota.WIKI,       # ← NOVO v4
}

_ESTADO_PARA_ROTA: dict[EstadoMenu, Rota] = {
    EstadoMenu.SUB_CALENDARIO: Rota.CALENDARIO,
    EstadoMenu.SUB_EDITAL:     Rota.EDITAL,
    EstadoMenu.SUB_CONTATOS:   Rota.CONTATOS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tipo de resultado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResultadoRoteamento:
    rota:       Rota
    tool_name:  str | None = None
    score:      float      = 0.0
    confianca:  str        = "baixa"     # "alta" | "media" | "baixa"
    metodo:     str        = "semantico" # "semantico" | "estado_menu" | "fallback_regex"

    @property
    def usar_tool_diretamente(self) -> bool:
        return self.confianca == "alta"


# ─────────────────────────────────────────────────────────────────────────────
# Roteamento
# ─────────────────────────────────────────────────────────────────────────────

def rotear(
    texto: str,
    estado_menu: EstadoMenu = EstadoMenu.MAIN,
) -> ResultadoRoteamento:
    """
    Determina a Rota e Tool mais adequadas para o texto dado.

    Ordem de decisão:
      1. Estado de submenu activo → força rota
      2. Busca vectorial KNN no Redis → tool mais similar
      3. Fallback para domain/router.py (regex)
    """
    # ── Rota forçada por submenu ──────────────────────────────────────────────
    if estado_menu in _ESTADO_PARA_ROTA:
        rota = _ESTADO_PARA_ROTA[estado_menu]
        return ResultadoRoteamento(rota=rota, score=1.0, confianca="alta", metodo="estado_menu")

    # ── Roteamento semântico ──────────────────────────────────────────────────
    try:
        resultado = _busca_tool_semantica(texto)
        if resultado:
            return resultado
    except Exception as e:
        logger.warning("⚠️  Roteamento semântico falhou: %s", e)

    # ── Fallback regex ────────────────────────────────────────────────────────
    return _fallback_regex(texto, estado_menu)


def _busca_tool_semantica(texto: str) -> ResultadoRoteamento | None:
    """KNN no Redis: retorna a tool com maior similaridade ao texto."""
    try:
        from src.rag.embeddings import get_embeddings
        vetor_bytes = _float_list_to_bytes(get_embeddings().embed_query(texto))
    except Exception as e:
        logger.warning("⚠️  Embedding para routing falhou: %s", e)
        return None

    r = get_redis()
    try:
        r.ft(IDX_TOOLS).info()
    except Exception:
        logger.debug("ℹ️  Índice IDX_TOOLS não existe ainda.")
        return None

    try:
        from redis.commands.search.query import Query as RQuery
        q = (
            RQuery("*=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("name", "score")
            .dialect(2)
        )
        results = r.ft(IDX_TOOLS).search(q, query_params={"vec": vetor_bytes})
    except Exception as e:
        logger.warning("⚠️  KNN tools falhou: %s", e)
        return None

    if not results.docs:
        return None

    doc        = results.docs[0]
    tool_name  = getattr(doc, "name", "")
    similarity = 1.0 - float(getattr(doc, "score", 1.0))

    logger.debug("🎯 Top tool: '%s' | sim=%.4f", tool_name, similarity)

    if similarity < THRESHOLD_MINIMO:
        return ResultadoRoteamento(rota=Rota.GERAL, score=similarity, confianca="baixa")

    rota = _TOOL_PARA_ROTA.get(tool_name, Rota.GERAL)
    if similarity >= THRESHOLD_ALTA:
        confianca = "alta"
    elif similarity >= THRESHOLD_MEDIA:
        confianca = "media"
    else:
        confianca = "baixa"
        rota = Rota.GERAL

    return ResultadoRoteamento(
        rota=rota, tool_name=tool_name, score=similarity,
        confianca=confianca, metodo="semantico",
    )


def _fallback_regex(texto: str, estado_menu: EstadoMenu) -> ResultadoRoteamento:
    """Fallback para domain/router.py quando Redis está offline."""
    try:
        from src.domain.router import analisar
        rota = analisar(texto, estado_menu)
        return ResultadoRoteamento(rota=rota, score=0.0, confianca="media", metodo="fallback_regex")
    except Exception as e:
        logger.error("❌ Fallback regex falhou: %s", e)
        return ResultadoRoteamento(rota=Rota.GERAL, score=0.0, confianca="baixa", metodo="fallback_regex")


# ─────────────────────────────────────────────────────────────────────────────
# Registo de tools no startup
# ─────────────────────────────────────────────────────────────────────────────

def registar_tools(tools: list) -> None:
    """
    Computa embeddings das descrições das tools e armazena no Redis.
    Chamado pelo AgentCore.inicializar() no startup.
    """
    try:
        from src.rag.embeddings import get_embeddings
        emb = get_embeddings()
        r   = get_redis()
        registadas = 0

        for tool in tools:
            try:
                name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
                desc = getattr(tool, "description", "") or ""
                if not name or not desc:
                    continue

                vetor = emb.embed_query(desc[:512])
                key   = f"{PREFIX_TOOLS}{name}"
                r.json().set(key, "$", {
                    "name":        name,
                    "description": desc[:512],
                    "embedding":   vetor,
                })
                registadas += 1
            except Exception as e:
                logger.warning("⚠️  Falha ao registar tool '%s': %s", tool, e)

        logger.info("🗺️  SemanticRouter: %d tools registadas no Redis.", registadas)

    except Exception as e:
        logger.error("❌ registar_tools falhou: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnóstico
# ─────────────────────────────────────────────────────────────────────────────

def listar_tools_registadas() -> list[dict]:
    r = get_redis()
    tools, cursor = [], 0
    while True:
        cursor, keys = r.scan(cursor, match=f"{PREFIX_TOOLS}*", count=100)
        for key in keys:
            try:
                doc = r.json().get(key, "$")
                if doc:
                    item = doc[0] if isinstance(doc, list) else doc
                    tools.append({
                        "name":        item.get("name", "?"),
                        "description": item.get("description", "?")[:80] + "...",
                    })
            except Exception:
                pass
        if cursor == 0:
            break
    return tools


def testar_roteamento(textos: list[str]) -> None:
    print("\n" + "=" * 65)
    print("🧪 TESTE DO ROTEAMENTO SEMÂNTICO v4")
    print("=" * 65)
    for texto in textos:
        r = rotear(texto)
        icone = {"alta": "✅", "media": "🔶", "baixa": "⚠️"}.get(r.confianca, "❓")
        print(f"{icone} [{r.rota.value:10s}] score={r.score:.3f} ({r.confianca:5s}) | '{texto[:50]}'")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Interno
# ─────────────────────────────────────────────────────────────────────────────

def _float_list_to_bytes(vetor: list[float]) -> bytes:
    return struct.pack(f"{len(vetor)}f", *vetor)