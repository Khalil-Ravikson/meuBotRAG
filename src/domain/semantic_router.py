"""
domain/semantic_router.py — Roteamento Semântico via Redis (Zero-LLM Routing)
===============================================================================

O QUE É O SEMANTIC TOOL ROUTING?
──────────────────────────────────
  Em vez de chamar o LLM para decidir qual tool usar ("preciso usar a tool de
  calendário ou a de edital?"), convertemos a pergunta em vetor e fazemos
  uma busca de similaridade nas descrições das tools armazenadas no Redis.

  ANTES (router.py com regex):
    - Mantém lista de padrões regex (frágil, precisa de manutenção manual)
    - Falha em variações linguísticas não previstas
    - Rápido, mas sem adaptabilidade

  AGORA (semantic_router.py com vetor):
    - Aprende semanticamente o que cada tool faz
    - Captura variações: "quando começa o semestre" = "início das aulas" = "data letivo"
    - Sem LLM → latência ~0.5ms (vs ~500ms do Groq/Gemini)
    - Sem tokens → custo $0

  COMPARAÇÃO DE CUSTO:
    10.000 mensagens/mês × 200 tokens de routing cada = 2.000.000 tokens/mês
    Groq free tier: esgota em ~3 dias
    Gemini free tier: esgota em ~2 dias
    Redis local: $0, 0ms de latência de rede

COMO FUNCIONA NA PRÁTICA:
──────────────────────────
  1. Na inicialização do bot:
     registrar_tools([tool_calendario, tool_edital, tool_contatos])
     → Embeddings das descrições são computados e armazenados no Redis

  2. Quando chega uma mensagem:
     rota = rotear("quando são as matrículas 2026?")
     → Pergunta é convertida em vetor (CPU local, ~5ms)
     → KNN no Redis encontra "consultar_calendario_academico" com score 0.92
     → Retorna Rota.CALENDARIO sem chamar nenhum LLM

  3. O roteamento semântico COMPLEMENTA o router de menu (domain/menu.py):
     - menu.py:            "1", "voltar", "oi" → navegação de menu (regex)
     - semantic_router.py: "onde fico minha prova?" → roteamento semântico

FALLBACK HIERÁRQUICO:
──────────────────────
  1. Threshold > 0.85 → confiança alta → usa a tool diretamente
  2. Threshold > 0.60 → confiança média → usa a tool mas consulta contexto extra
  3. Threshold < 0.60 → confiança baixa → passa para Rota.GERAL (LLM decide)
  4. Redis offline → fallback para router regex (domain/router.py)
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

from src.domain.entities import EstadoMenu, Rota
from src.infrastructure.redis_client import (
    IDX_TOOLS,
    PREFIX_TOOLS,
    VECTOR_DIM,
    get_redis,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração de thresholds
# ─────────────────────────────────────────────────────────────────────────────

# COSINE similarity: 1.0 = idêntico, 0.0 = ortogonal
# Para embeddings BAAI/bge-m3 (multilíngue), valores típicos:
#   > 0.80 → muito similar (mesma intenção)
#   0.65-0.80 → similar (provavelmente mesma área)
#   < 0.65 → diferente (área incerta)
THRESHOLD_ALTA    = 0.80    # Usa tool com alta confiança
THRESHOLD_MEDIA   = 0.62    # Usa tool mas sinaliza incerteza
THRESHOLD_MINIMO  = 0.40    # Abaixo disso → Rota.GERAL


# ─────────────────────────────────────────────────────────────────────────────
# Mapeamento tool_name → Rota
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_PARA_ROTA: dict[str, Rota] = {
    "consultar_calendario_academico": Rota.CALENDARIO,
    "consultar_edital_paes_2026":     Rota.EDITAL,
    "consultar_contatos_uema":        Rota.CONTATOS,
}

# Rota forçada quando o utilizador está num submenu ativo
_ESTADO_PARA_ROTA: dict[EstadoMenu, Rota] = {
    EstadoMenu.SUB_CALENDARIO: Rota.CALENDARIO,
    EstadoMenu.SUB_EDITAL:     Rota.EDITAL,
    EstadoMenu.SUB_CONTATOS:   Rota.CONTATOS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tipos de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResultadoRoteamento:
    """Resultado do roteamento semântico com métricas de confiança."""
    rota:          Rota
    tool_name:     str | None   = None
    score:         float        = 0.0
    confianca:     str          = "baixa"    # "alta" | "media" | "baixa"
    metodo:        str          = "semantico" # "semantico" | "estado_menu" | "fallback_regex"

    @property
    def usar_tool_diretamente(self) -> bool:
        """Alta confiança → passa o contexto já pré-filtrado para a tool."""
        return self.confianca == "alta"


# ─────────────────────────────────────────────────────────────────────────────
# Registação das tools no Redis
# ─────────────────────────────────────────────────────────────────────────────

def registar_tools(tools: list) -> None:
    """
    Regista as tools no Redis com os seus embeddings.

    CORRIGIDO: Usamos get_redis_text() em vez de get_redis() para as
    operações JSON. O cliente com decode_responses=False (bytes) pode
    causar inconsistências ao ler campos de string como 'name' e
    'description' via r.json().get() — o RedisJSON devolve bytes que
    depois não são comparáveis com strings Python.

    O embedding (lista de floats) é serializado como JSON nativo pelo
    RedisJSON em ambos os modos, mas a leitura posterior dos campos de
    texto é mais fiável com decode_responses=True.

    NOTA: O índice IDX_TOOLS foi criado com o cliente de bytes em
    criar_indice_tools(), o que está correto — a criação de índice não
    lê dados. Apenas a escrita/leitura de documentos JSON beneficia
    do cliente de texto.
    """
    from src.rag.embeddings import get_embeddings
    from src.infrastructure.redis_client import get_redis_text, PREFIX_TOOLS

    embeddings_model = get_embeddings()

    # CORRIGIDO: get_redis_text() para operações de leitura/escrita JSON
    r = get_redis_text()

    registadas = 0
    for tool in tools:
        name = getattr(tool, "name", None)
        desc = getattr(tool, "description", None)

        if not name or not desc:
            logger.warning("⚠️  Tool sem name/description ignorada: %s", tool)
            continue

        # Texto rico para embedding = nome + descrição completa
        texto_embedding = f"{name}: {desc}"

        try:
            # Computa embedding (CPU local, ~20ms por tool, modelo já em memória)
            vetor: list[float] = embeddings_model.embed_query(texto_embedding)

            key = f"{PREFIX_TOOLS}{name}"

            # Armazena no Redis como RedisJSON
            # O RedisJSON aceita listas Python nativas como arrays JSON
            r.json().set(key, "$", {
                "name":        name,
                "description": desc,
                "embedding":   vetor,   # lista[float] → JSON array
            })

            registadas += 1
            logger.debug("📌 Tool registada: '%s' (embedding dim=%d)", name, len(vetor))

        except Exception as e:
            logger.error("❌ Falha ao registar tool '%s': %s", name, e)

    logger.info(
        "✅ %d/%d tools registadas no Redis para semantic routing.",
        registadas, len(tools),
    )
# ─────────────────────────────────────────────────────────────────────────────
# Roteamento
# ─────────────────────────────────────────────────────────────────────────────

def rotear(
    texto: str,
    estado_menu: EstadoMenu = EstadoMenu.MAIN,
) -> ResultadoRoteamento:
    """
    Determina a Rota e Tool mais adequadas para o texto dado.

    PIPELINE DE DECISÃO:
    ────────────────────
      1. Estado do submenu ativo → força rota (submenu de calendário = CALENDARIO)
      2. Busca vetorial no Redis → encontra tool mais similar
      3. Threshold decision:
           score > THRESHOLD_ALTA  → confiança alta, usa tool diretamente
           score > THRESHOLD_MEDIA → confiança média, usa tool com cuidado
           score < THRESHOLD_MINIMO → fallback para router regex
      4. Se Redis offline → fallback para domain/router.py (regex)

    Parâmetros:
      texto:       Mensagem do utilizador (ou prompt expandido do menu)
      estado_menu: Estado atual do menu (para força de rota por submenu)
    """
    # ── 1. Rota forçada pelo submenu ──────────────────────────────────────────
    if estado_menu in _ESTADO_PARA_ROTA:
        rota_forcada = _ESTADO_PARA_ROTA[estado_menu]
        logger.debug("📌 Rota forçada por submenu %s → %s", estado_menu.value, rota_forcada.value)
        return ResultadoRoteamento(
            rota=rota_forcada,
            score=1.0,
            confianca="alta",
            metodo="estado_menu",
        )

    # ── 2. Roteamento semântico via Redis ─────────────────────────────────────
    try:
        resultado_semantico = _busca_tool_semantica(texto)
        if resultado_semantico:
            return resultado_semantico
    except Exception as e:
        logger.warning("⚠️  Roteamento semântico falhou, usando regex: %s", e)

    # ── 3. Fallback para router regex (domain/router.py) ─────────────────────
    return _fallback_regex(texto, estado_menu)


def _busca_tool_semantica(texto: str) -> ResultadoRoteamento | None:
    """
    Busca vetorial KNN no Redis para encontrar a tool mais similar.

    COMO FUNCIONA O KNN NO REDIS:
    ──────────────────────────────
      Query: "*=>[KNN 1 @embedding $vec AS score]"
        - "*"                → sem pré-filtro (temos poucas tools, busca todas)
        - KNN 1              → retorna apenas o top-1 resultado
        - @embedding         → campo vetorial do esquema
        - $vec               → parâmetro da query (bytes do vetor)
        - AS score           → score de distância como campo retornado

      NOTA SOBRE O SCORE DO REDIS:
        Redis retorna DISTÂNCIA cosine (0 = idêntico, 2 = oposto).
        Convertemos para SIMILARIDADE: similarity = 1 - (distance / 2)
        Normalizado: 1.0 = idêntico, 0.0 = completamente diferente
    """
    from src.rag.embeddings import get_embeddings  # Import local
    from redis.commands.search.query import Query

    embeddings_model = get_embeddings()
    r = get_redis()

    # Verifica se o índice existe e tem dados
    try:
        info = r.ft(IDX_TOOLS).info()
        if info.get("num_docs", 0) == 0:
            logger.debug("⚠️  Índice de tools vazio, pulando roteamento semântico.")
            return None
    except Exception:
        return None

    # Converte texto para vetor
    vetor = embeddings_model.embed_query(texto)

    # Serializa como bytes FLOAT32
    embedding_bytes = struct.pack(f"{len(vetor)}f", *vetor)

    # Query KNN — busca top-3 para ter alternativas
    query = (
        Query("*=>[KNN 3 @embedding $vec AS vec_dist]")
        .sort_by("vec_dist")
        .return_fields("name", "description", "vec_dist")
        .dialect(2)
        .paging(0, 3)
    )

    resultados = r.ft(IDX_TOOLS).search(query, {"vec": embedding_bytes})

    if not resultados.docs:
        return None

    # Melhor resultado (top-1)
    top = resultados.docs[0]
    tool_name = getattr(top, "name", None)
    # Redis retorna distância cosine como bytes ou string
    raw_dist  = getattr(top, "vec_dist", None)

    if not tool_name or raw_dist is None:
        return None

    # Converte distância para similaridade [0, 1]
    try:
        dist = float(raw_dist)
        similarity = max(0.0, 1.0 - (dist / 2.0))
    except (ValueError, TypeError):
        return None

    logger.debug(
        "🎯 Semantic routing | texto='%.40s' → tool='%s' | score=%.3f",
        texto, tool_name, similarity,
    )

    # Determina confiança baseada no threshold
    if similarity >= THRESHOLD_ALTA:
        confianca = "alta"
    elif similarity >= THRESHOLD_MEDIA:
        confianca = "media"
    elif similarity >= THRESHOLD_MINIMO:
        confianca = "baixa"
    else:
        # Abaixo do mínimo → nenhuma tool é adequada → Rota.GERAL
        logger.debug("⚠️  Score %.3f abaixo do mínimo %.3f → Rota.GERAL", similarity, THRESHOLD_MINIMO)
        return ResultadoRoteamento(
            rota=Rota.GERAL,
            tool_name=None,
            score=similarity,
            confianca="baixa",
            metodo="semantico",
        )

    rota = _TOOL_PARA_ROTA.get(tool_name, Rota.GERAL)

    return ResultadoRoteamento(
        rota=rota,
        tool_name=tool_name,
        score=similarity,
        confianca=confianca,
        metodo="semantico",
    )


def _fallback_regex(texto: str, estado_menu: EstadoMenu) -> ResultadoRoteamento:
    """
    Fallback para o router regex existente (domain/router.py).
    Usado quando Redis está offline ou sem tools registadas.

    PRESERVA COMPATIBILIDADE:
      O router regex funciona bem para os casos cobertos pelos padrões.
      O roteamento semântico é um UPGRADE, não uma substituição total.
    """
    try:
        from src.domain.router import analisar  # Import local para evitar circular
        rota = analisar(texto, estado_menu)
        logger.debug("📋 Fallback regex | rota=%s", rota.value)
        return ResultadoRoteamento(
            rota=rota,
            score=0.0,
            confianca="media",
            metodo="fallback_regex",
        )
    except Exception as e:
        logger.error("❌ Fallback regex falhou: %s", e)
        return ResultadoRoteamento(
            rota=Rota.GERAL,
            score=0.0,
            confianca="baixa",
            metodo="fallback_regex",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Funções de utilidade e diagnóstico
# ─────────────────────────────────────────────────────────────────────────────

def listar_tools_registadas() -> list[dict]:
    """
    Lista todas as tools registadas no Redis.
    Útil para /banco/sources e debug.
    """
    r = get_redis()
    tools = []
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match=f"{PREFIX_TOOLS}*", count=100)
        for key in keys:
            try:
                doc = r.json().get(key, "$")
                if doc:
                    item = doc[0] if isinstance(doc, list) else doc
                    tools.append({
                        "name":        item.get("name", "?"),
                        "description": item.get("description", "?")[:100] + "...",
                    })
            except Exception:
                pass
        if cursor == 0:
            break
    return tools


def testar_roteamento(textos: list[str]) -> None:
    """
    Testa o roteamento para uma lista de textos.
    Útil durante desenvolvimento para validar os thresholds.

    Uso:
      from src.domain.semantic_router import testar_roteamento
      testar_roteamento([
          "quando é a matrícula?",
          "quantas vagas tem engenharia civil?",
          "email da pró-reitoria",
          "oi bom dia",
      ])
    """
    print("\n" + "=" * 65)
    print("🧪 TESTE DO ROTEAMENTO SEMÂNTICO")
    print("=" * 65)

    for texto in textos:
        resultado = rotear(texto)
        icone = {"alta": "✅", "media": "🔶", "baixa": "⚠️"}.get(resultado.confianca, "❓")
        print(
            f"{icone} [{resultado.rota.value:10s}] "
            f"score={resultado.score:.3f} "
            f"({resultado.confianca:5s}) "
            f"metodo={resultado.metodo:15s} | "
            f"'{texto[:45]}'"
        )

    print("=" * 65 + "\n")