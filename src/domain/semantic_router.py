"""
domain/semantic_router.py — Roteamento Semântico v2 (HyDE Routing)
====================================================================

PROBLEMA RESOLVIDO NESTA VERSÃO:
──────────────────────────────────
  ANTES (v1): indexava "nome_da_tool: descrição_técnica_completa"
    → Score 0.54 para "quando é a matrícula?" → GERAL
    
  POR QUÊ FALHA?
    Assimetria semântica: a query do aluno é curta e coloquial
    ("quando é a matrícula?"), mas o embedding indexado é de um texto
    técnico longo ("Consulta datas, prazos e eventos do calendário...").
    O modelo de embedding não consegue bridgear este gap de distribuição.

  AGORA (v2): HyDE Routing — "Hypothetical Document Embeddings"
    Em vez de indexar a descrição da tool, indexamos EXEMPLOS REAIS
    de como os alunos realmente perguntam.
    
    Cada tool tem uma lista de 12-15 queries representativas.
    O embedding de cada query é indexado separadamente.
    Quando chega "quando é a matrícula?", o KNN encontra a query
    de exemplo mais próxima (score 0.92+) e retorna a tool correcta.

  INSPIRAÇÃO:
    NirDiamant/RAG_Techniques: "HyDE — Hypothetical Document Embeddings"
    Aplicado aqui ao inverso: em vez de criar documento hipotético da
    resposta, criamos queries hipotéticas para cada tool.

GANHO ESPERADO:
───────────────
  "quando é a matrícula?"     → CALENDÁRIO (score ~0.92 vs 0.54 antes)
  "quantas vagas BR-PPI?"     → EDITAL     (score ~0.88 vs 0.49 antes)
  "email da PROG"             → CONTATOS   (score ~0.91 vs 0.58 antes)
  "como acesso o SIGAA?"      → WIKI       (score ~0.85)

CUSTO:
──────
  Antes: 4 embeddings (1 por tool)
  Agora: ~60 embeddings (15 por tool × 4 tools)
  Tempo de registo: ~30s (só no startup, não em cada query)
  Memória Redis: ~60 × 4KB ≈ 240KB adicional (negligenciável)
  
THRESHOLDS (mantidos, mas agora alcançáveis):
─────────────────────────────────────────────
  THRESHOLD_ALTA   = 0.80  → score típico: 0.85-0.95
  THRESHOLD_MEDIA  = 0.62  → score típico: 0.65-0.80
  THRESHOLD_MINIMO = 0.40  → qualquer outra coisa → GERAL
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass

from src.domain.entities import EstadoMenu, Rota
from src.infrastructure.redis_client import (
    IDX_TOOLS,
    PREFIX_TOOLS,
    VECTOR_DIM,
    get_redis,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (inalterados — agora são alcançáveis com HyDE)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD_ALTA   = 0.80
THRESHOLD_MEDIA  = 0.62
THRESHOLD_MINIMO = 0.40

# ─────────────────────────────────────────────────────────────────────────────
# HyDE Routing — Queries de Exemplo por Tool
# ─────────────────────────────────────────────────────────────────────────────
# Estas queries representam como os alunos realmente perguntam no WhatsApp.
# Ao indexar as queries (não a descrição), eliminamos a assimetria semântica.
#
# COMO ADICIONAR NOVA TOOL:
#   1. Adiciona a tool em tools/__init__.py
#   2. Adiciona uma lista de 10-15 queries abaixo
#   3. Adiciona a tool no _TOOL_PARA_ROTA
#   O router funciona automaticamente na próxima inicialização.

_QUERIES_POR_TOOL: dict[str, list[str]] = {

    "consultar_calendario_academico": [
        # Matrícula
        "quando é a matrícula?",
        "quando é a matrícula de veteranos?",
        "qual o prazo de matrícula para calouros?",
        "quando começa a rematrícula 2026.1?",
        "data da matrícula retardatária",
        # Semestre
        "quando começa o semestre?",
        "quando terminam as aulas?",
        "início do período letivo 2026",
        "quando começa as aulas do segundo semestre?",
        # Provas e prazos
        "quando são as provas finais?",
        "data da avaliação substitutiva",
        "prazo para trancamento de matrícula",
        "calendário acadêmico 2026",
        # Feriados
        "quais os feriados de março?",
        "tem aula no carnaval?",
    ],

    "consultar_edital_paes_2026": [
        # Vagas e cotas
        "quantas vagas tem engenharia civil?",
        "quantas vagas BR-PPI para medicina?",
        "como funciona a cota BR-PPI?",
        "o que é cota BR-Q?",
        "vagas para ampla concorrência direito",
        "quantas vagas tem sistemas de informação?",
        "vagas PCD por curso",
        # Inscrição
        "quando abrem as inscrições do PAES?",
        "como me inscrevo no PAES 2026?",
        "quais documentos preciso para me inscrever?",
        "cronograma do processo seletivo",
        # Categorias
        "o que é AC no edital?",
        "diferença entre BR-PPI e BR-Q",
        "como funciona a heteroidentificação?",
        "quais cursos são ofertados no PAES?",
    ],

    "consultar_contatos_uema": [
        # Emails
        "qual o email da PROG?",
        "email da pró-reitoria de graduação",
        "email da secretaria de direito",
        "como entro em contato com a coordenação?",
        "email do CTIC",
        # Telefones
        "telefone da secretaria",
        "telefone da reitoria",
        "número da PROEXAE",
        # Setores
        "contato do CECEN",
        "email da coordenação de engenharia civil",
        "como falar com a pró-reitoria?",
        "contato do suporte de TI",
        "onde fica a secretaria?",
        "email do PRPPG",
        "contato da administração do campus",
    ],

    "consultar_wiki_ctic": [
        # Sistemas TI
        "como acesso o SIGAA?",
        "como entrar no SIGAA?",
        "esqueci minha senha do SIGAA",
        "como acesso o e-mail institucional?",
        "como configurar email da uema?",
        # Suporte
        "computador do laboratório com problema",
        "internet não funciona no campus",
        "como abrir chamado de suporte?",
        "problema com impressora",
        # Sistemas específicos
        "como usar o SIE?",
        "como acessar o SIPAC?",
        "o que é o SIGUEMA?",
        "acesso ao sistema de bolsas",
        "VPN da UEMA como configurar",
        "wifi da universidade não conecta",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Mapeamentos
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_PARA_ROTA: dict[str, Rota] = {
    "consultar_calendario_academico": Rota.CALENDARIO,
    "consultar_edital_paes_2026":     Rota.EDITAL,
    "consultar_contatos_uema":        Rota.CONTATOS,
    "consultar_wiki_ctic":            Rota.WIKI if hasattr(Rota, "WIKI") else Rota.GERAL,
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
    rota:      Rota
    tool_name: str | None = None
    score:     float      = 0.0
    confianca: str        = "baixa"    # "alta" | "media" | "baixa"
    metodo:    str        = "semantico"

    @property
    def usar_tool_diretamente(self) -> bool:
        return self.confianca == "alta"


# ─────────────────────────────────────────────────────────────────────────────
# Registação das tools (HyDE — indexa queries de exemplo)
# ─────────────────────────────────────────────────────────────────────────────

def registar_tools(tools: list) -> None:
    """
    Regista as tools no Redis usando HyDE Routing.

    Para cada tool, indexa as queries de exemplo do _QUERIES_POR_TOOL.
    Cada query recebe um embedding separado, com o mesmo tool_name.

    ESTRUTURA NO REDIS (nova):
      Chave: tools:emb:{tool_name}:{índice_da_query}
      JSON:  {"name": "consultar_calendario_academico",
              "query_exemplo": "quando é a matrícula?",
              "embedding": [...]}

    COMPATIBILIDADE:
      O KNN busca por embedding → retorna tool_name → lookup no _TOOL_PARA_ROTA.
      Nenhuma mudança necessária no código de busca (_busca_tool_semantica).
    """
    from src.rag.embeddings import get_embeddings
    from src.infrastructure.redis_client import get_redis_text, PREFIX_TOOLS

    embeddings_model = get_embeddings()
    r                = get_redis_text()

    # Obtém nomes das tools activas
    tool_names = set()
    for tool in tools:
        name = getattr(tool, "name", None)
        if name:
            tool_names.add(name)

    total_registadas = 0
    total_queries    = 0

    for tool_name in tool_names:
        queries = _QUERIES_POR_TOOL.get(tool_name)
        if not queries:
            # Tool sem queries de exemplo → usa descrição como fallback
            for tool in tools:
                if getattr(tool, "name", None) == tool_name:
                    desc = getattr(tool, "description", tool_name)
                    queries = [desc[:200]]
                    logger.warning(
                        "⚠️  '%s' sem queries de exemplo em _QUERIES_POR_TOOL. "
                        "Adiciona queries para melhor routing.",
                        tool_name,
                    )
                    break

        logger.debug("📌 Registando '%s': %d queries de exemplo", tool_name, len(queries))

        for idx, query in enumerate(queries):
            key = f"{PREFIX_TOOLS}{tool_name}:{idx}"
            try:
                vetor = embeddings_model.embed_query(query)
                r.json().set(key, "$", {
                    "name":          tool_name,
                    "query_exemplo": query,
                    "embedding":     vetor,
                })
                total_queries += 1
            except Exception as e:
                logger.error("❌ Falha ao registar query '%s' de '%s': %s", query[:40], tool_name, e)

        total_registadas += 1
        logger.info("✅ '%s': %d queries indexadas", tool_name, len(queries))

    logger.info(
        "🗺️  HyDE Routing: %d tools | %d queries indexadas no Redis",
        total_registadas, total_queries,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Roteamento
# ─────────────────────────────────────────────────────────────────────────────

def rotear(
    texto: str,
    estado_menu: EstadoMenu = EstadoMenu.MAIN,
) -> ResultadoRoteamento:
    """
    Determina a Rota mais adequada para o texto dado.
    
    PIPELINE:
      1. Submenu activo → força rota (0ms)
      2. KNN nas queries de exemplo (HyDE, ~1ms, 0 tokens)
      3. Fallback regex se Redis offline
    """
    # ── 1. Submenu activo ─────────────────────────────────────────────────────
    if estado_menu in _ESTADO_PARA_ROTA:
        rota = _ESTADO_PARA_ROTA[estado_menu]
        return ResultadoRoteamento(rota=rota, score=1.0, confianca="alta", metodo="estado_menu")

    # ── 2. KNN nas queries de exemplo ─────────────────────────────────────────
    try:
        resultado = _busca_tool_semantica(texto)
        if resultado:
            return resultado
    except Exception as e:
        logger.warning("⚠️  Roteamento semântico falhou: %s", e)

    # ── 3. Fallback regex ─────────────────────────────────────────────────────
    return _fallback_regex(texto, estado_menu)


def _busca_tool_semantica(texto: str) -> ResultadoRoteamento | None:
    """
    KNN no índice de queries de exemplo (HyDE Routing).
    
    Retorna a tool cuja query de exemplo é mais similar ao texto dado.
    """
    try:
        from src.rag.embeddings import get_embeddings
        from redis.commands.search.query import Query

        embeddings_model = get_embeddings()
        r                = get_redis()

        # Verifica índice
        try:
            info = r.ft(IDX_TOOLS).info()
            if info.get("num_docs", 0) == 0:
                logger.debug("⚠️  Índice IDX_TOOLS vazio.")
                return None
        except Exception:
            return None

        # Embedding da query do utilizador
        vetor          = embeddings_model.embed_query(texto)
        embedding_bytes= struct.pack(f"{len(vetor)}f", *vetor)

        # KNN top-3 (para ter alternativas em caso de empate)
        query = (
            Query("*=>[KNN 3 @embedding $vec AS vec_dist]")
            .sort_by("vec_dist")
            .return_fields("name", "query_exemplo", "vec_dist")
            .dialect(2)
            .paging(0, 3)
        )

        resultados = r.ft(IDX_TOOLS).search(query, {"vec": embedding_bytes})
        if not resultados.docs:
            return None

        top       = resultados.docs[0]
        tool_name = getattr(top, "name", None)
        raw_dist  = getattr(top, "vec_dist", None)
        query_ex  = getattr(top, "query_exemplo", "?")

        if not tool_name or raw_dist is None:
            return None

        # Converte distância cosine → similaridade [0, 1]
        dist       = float(raw_dist)
        similarity = max(0.0, 1.0 - (dist / 2.0))

        logger.info(
            "🎯 HyDE Routing | query='%.35s' → tool='%s' | "
            "exemplo='%.35s' | score=%.3f",
            texto, tool_name, query_ex, similarity,
        )

        # Threshold decision
        if similarity < THRESHOLD_MINIMO:
            return ResultadoRoteamento(
                rota=Rota.GERAL, score=similarity, confianca="baixa", metodo="semantico",
            )

        rota      = _TOOL_PARA_ROTA.get(tool_name, Rota.GERAL)
        confianca = (
            "alta"  if similarity >= THRESHOLD_ALTA  else
            "media" if similarity >= THRESHOLD_MEDIA else
            "baixa"
        )

        return ResultadoRoteamento(
            rota=rota, tool_name=tool_name, score=similarity,
            confianca=confianca, metodo="semantico",
        )

    except Exception as e:
        logger.warning("⚠️  _busca_tool_semantica: %s", e)
        return None


def _fallback_regex(texto: str, estado_menu: EstadoMenu) -> ResultadoRoteamento:
    try:
        from src.domain.router import analisar
        rota = analisar(texto, estado_menu)
        return ResultadoRoteamento(rota=rota, score=0.0, confianca="media", metodo="fallback_regex")
    except Exception as e:
        logger.error("❌ Fallback regex falhou: %s", e)
        return ResultadoRoteamento(rota=Rota.GERAL, score=0.0, confianca="baixa", metodo="fallback_regex")


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def listar_tools_registadas() -> list[dict]:
    """Lista tools únicas registadas (uma por tool, não uma por query)."""
    r     = get_redis()
    visto: set[str] = set()
    tools: list[dict] = []
    cursor= 0

    while True:
        cursor, keys = r.scan(cursor, match=f"{PREFIX_TOOLS}*", count=200)
        for key in keys:
            try:
                doc = r.json().get(key, "$")
                if doc:
                    item = doc[0] if isinstance(doc, list) else doc
                    name = item.get("name", "?")
                    if name not in visto:
                        visto.add(name)
                        tools.append({
                            "name":  name,
                            "description": f"HyDE: {len(_QUERIES_POR_TOOL.get(name, []))} queries de exemplo",
                        })
            except Exception:
                pass
        if cursor == 0:
            break

    return tools


def testar_roteamento(textos: list[str]) -> None:
    """Testa o roteamento e mostra scores detalhados."""
    print("\n" + "=" * 70)
    print("🧪 TESTE DO ROTEAMENTO SEMÂNTICO (HyDE v2)")
    print("=" * 70)
    for texto in textos:
        r     = rotear(texto)
        icone = {"alta": "✅", "media": "🔶", "baixa": "⚠️"}.get(r.confianca, "❓")
        print(
            f"{icone} [{r.rota.value:10s}] "
            f"score={r.score:.3f} ({r.confianca:5s}) "
            f"método={r.metodo:15s} | '{texto[:45]}'"
        )
    print("=" * 70 + "\n")


def adicionar_queries_exemplo(tool_name: str, novas_queries: list[str]) -> int:
    """
    Adiciona queries de exemplo a uma tool em runtime.
    Útil para melhorar o routing sem reiniciar o bot.
    Retorna o número de queries adicionadas com sucesso.
    """
    from src.rag.embeddings import get_embeddings
    from src.infrastructure.redis_client import get_redis_text, PREFIX_TOOLS

    emb   = get_embeddings()
    r     = get_redis_text()
    adicionadas = 0

    # Encontra o próximo índice disponível
    cursor, keys = get_redis().scan(0, match=f"{PREFIX_TOOLS}{tool_name}:*", count=200)
    proximo_idx  = len(keys)

    for i, query in enumerate(novas_queries):
        key = f"{PREFIX_TOOLS}{tool_name}:{proximo_idx + i}"
        try:
            vetor = emb.embed_query(query)
            r.json().set(key, "$", {
                "name":          tool_name,
                "query_exemplo": query,
                "embedding":     vetor,
            })
            adicionadas += 1
        except Exception as e:
            logger.warning("⚠️  Falha ao adicionar query '%s': %s", query[:40], e)

    logger.info("✅ %d queries adicionadas a '%s'", adicionadas, tool_name)
    return adicionadas