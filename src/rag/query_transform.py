"""
rag/query_transform.py — Transformação de Queries com Contexto Factual
========================================================================

O PROBLEMA QUE ESTA CAMADA RESOLVE:
──────────────────────────────────────
  As perguntas reais dos alunos são vagas, coloquiais e sem contexto:

  ❌ "quando é minha prova?" 
     → Busca retorna: tudo sobre provas em todos os cursos
  
  ❌ "quantas vagas?" 
     → Busca retorna: vagas de todos os 80+ cursos do edital
  
  ❌ "posso trancar?"
     → Busca retorna: regras gerais de trancamento (pode alucinações em datas)

  Com Query Transformation + Fatos:

  ✅ "quando é minha prova?" + fato "Aluno de Engenharia Civil, noturno"
     → Transformed: "avaliação final Engenharia Civil turno noturno 2026.1 data local"
     → Busca híbrida encontra o chunk EXATO

  ✅ "quantas vagas?" + fato "Interesse em Sistemas de Informação"
     → Transformed: "vagas Sistemas de Informação AC PcD BR-PPI PAES 2026"
     → Busca encontra a linha exacta da tabela do edital

TÉCNICAS IMPLEMENTADAS (inspiradas no RAG Techniques de Nir Diamant):
───────────────────────────────────────────────────────────────────────

  1. QUERY REWRITING (principal)
     Reformula com termos técnicos + contexto factual
     Custo: 1 chamada ao Gemini, ~80 tokens input + ~40 output
     Redução de tokens na busca final: ~60% (query mais precisa → menos chunks)

  2. SUB-QUERY DECOMPOSITION (para perguntas complexas)
     "Quais são os documentos e datas para me inscrever no PAES como cotista?"
     → Sub-query 1: "documentos necessários inscrição PAES 2026"
     → Sub-query 2: "cronograma datas inscrição PAES 2026"
     → Sub-query 3: "categorias cotas PAES 2026 requisitos"
     Cada sub-query busca de forma independente e os resultados são fundidos.

  3. STEP-BACK PROMPTING (para perguntas muito específicas)
     "Qual é o email da coordenadora do curso de Engenharia Civil?"
     → Step-back: "contatos coordenação Engenharia Civil UEMA"
     → Busca mais ampla que captura mesmo se o cargo mudou de pessoa

CUSTO TOTAL DA CAMADA:
───────────────────────
  Chamada Gemini: ~120 tokens (80 input + 40 output)
  vs. custo de não transformar: busca retorna chunks errados → 2ª tentativa
  → Economiza em média 1 chamada de geração ao Gemini por conversa

QUANDO NÃO TRANSFORMAR:
────────────────────────
  A função _precisa_transformar() detecta queries já técnicas:
  - "matrícula veteranos 2026.1" → já técnica, não transforma
  - "email PROG pró-reitoria" → já técnica, não transforma
  Isso evita chamar o Gemini quando desnecessário.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.memory.long_term_memory import Fato, fatos_como_string
from src.providers.gemini_provider import (
    PROMPT_QUERY_REWRITE,
    chamar_gemini_estruturado,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Perguntas mais curtas que este limite provavelmente não precisam de reescrita
_MIN_CHARS_PARA_TRANSFORM = 15

# Palavras que indicam que a query já é técnica o suficiente
_TERMOS_JA_TECNICOS = frozenset({
    "matricula", "rematricula", "trancamento", "edital", "paes",
    "ac", "pcd", "br-ppi", "br-q", "br-dc", "ir-ppi",
    "prog", "proexae", "prppg", "prad", "ctic",
    "cecen", "cesb", "cesc", "ccsa",
    "2026.1", "2026.2", "calendario",
    "cronograma", "inscricao", "documentos",
})

# Comprimento máximo da query transformada (para não desperdiçar tokens)
_MAX_QUERY_CHARS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryTransformada:
    """
    Resultado da transformação de query com metadados.

    Contém tanto a query principal reescrita como sub-queries
    opcionais para perguntas complexas.
    """
    query_original:   str
    query_principal:  str
    sub_queries:      list[str] = field(default_factory=list)
    palavras_chave:   list[str] = field(default_factory=list)
    foi_transformada: bool = False
    motivo:           str = ""

    @property
    def todas_queries(self) -> list[str]:
        """Retorna query principal + sub-queries para busca paralela."""
        queries = [self.query_principal]
        queries.extend(self.sub_queries)
        return queries

    @property
    def query_para_log(self) -> str:
        arrow = " → " if self.foi_transformada else " (sem transform)"
        return f"'{self.query_original[:40]}'{arrow}'{self.query_principal[:60]}'"


# ─────────────────────────────────────────────────────────────────────────────
# API principal
# ─────────────────────────────────────────────────────────────────────────────

def transformar_query(
    pergunta: str,
    fatos_usuario: list[Fato] | None = None,
    usar_sub_queries: bool = False,
) -> QueryTransformada:
    """
    Transforma a pergunta do utilizador numa query otimizada para busca.

    FLUXO DE DECISÃO:
    ──────────────────
      1. A query já é técnica? → retorna sem transformar (economiza chamada Gemini)
      2. A query é muito curta? → enriquece com fatos mas sem LLM
      3. Query normal → chama Gemini com fatos como contexto
      4. Falha do Gemini? → usa a query original (graceful degradation)

    Parâmetros:
      pergunta:       Texto original do utilizador
      fatos_usuario:  Fatos relevantes da Long-Term Memory
      usar_sub_queries: Se True, gera sub-queries para perguntas complexas
    """
    pergunta_limpa = pergunta.strip()

    if not pergunta_limpa:
        return QueryTransformada(
            query_original=pergunta,
            query_principal=pergunta,
            motivo="pergunta vazia",
        )

    # ── 1. Verifica se já é técnica o suficiente ─────────────────────────────
    if not _precisa_transformar(pergunta_limpa, fatos_usuario):
        logger.debug("⚡ Query já técnica, sem transform: '%.50s'", pergunta_limpa)
        return QueryTransformada(
            query_original=pergunta_limpa,
            query_principal=pergunta_limpa,
            foi_transformada=False,
            motivo="query já técnica",
        )

    # ── 2. Monta texto dos fatos para o prompt ────────────────────────────────
    fatos_str = fatos_como_string(fatos_usuario) if fatos_usuario else "(sem histórico do aluno)"

    # ── 3. Decide a técnica de transformação ──────────────────────────────────
    if usar_sub_queries and _e_pergunta_complexa(pergunta_limpa):
        return _transformar_com_sub_queries(pergunta_limpa, fatos_str)
    else:
        return _transformar_query_simples(pergunta_limpa, fatos_str)


def transformar_para_step_back(pergunta: str) -> str:
    """
    Step-Back Prompting: gera uma versão mais genérica da pergunta.

    QUANDO USAR:
      Quando a busca híbrida retorna 0 resultados, tentamos uma versão
      mais ampla da pergunta para capturar informação mesmo que parcial.

    Exemplo:
      "qual é o email da Prof. Maria coordenadora de Engenharia Civil?"
      → step-back: "contato coordenação Engenharia Civil UEMA"

    Esta versão é SÍNCRONA e não usa o Gemini — usa heurísticas simples
    para ser rápida e economizar tokens no caminho de fallback.
    """
    # Remove nomes próprios (heurística simples)
    pergunta_sem_nomes = re.sub(
        r'\b(prof\.?|professora?|dr\.?|doutora?)\s+\w+', '', pergunta, flags=re.IGNORECASE
    )

    # Remove artigos e preposições iniciais
    pergunta_sem_nomes = re.sub(r'^(qual é|qual o|qual a|onde fica|como é)\s+', '', pergunta_sem_nomes, flags=re.IGNORECASE)

    # Remove detalhes muito específicos (números, datas)
    pergunta_sem_detalhes = re.sub(r'\b\d{4,}\b', '', pergunta_sem_nomes)

    resultado = pergunta_sem_detalhes.strip()
    return resultado if len(resultado) > 10 else pergunta


# ─────────────────────────────────────────────────────────────────────────────
# Transformações internas
# ─────────────────────────────────────────────────────────────────────────────

def _transformar_query_simples(pergunta: str, fatos_str: str) -> QueryTransformada:
    """
    Transformação simples: reescreve a query com termos técnicos.
    Usa o prompt PROMPT_QUERY_REWRITE do gemini_provider.
    """
    prompt = PROMPT_QUERY_REWRITE.format(
        fatos=fatos_str,
        pergunta=pergunta,
    )

    resultado = chamar_gemini_estruturado(
        prompt=prompt,
        schema_descricao='{"query_reescrita": "string", "palavras_chave": ["string"]}',
        temperatura=0.1,
    )

    if not resultado:
        logger.warning("⚠️  Query transform falhou, usando original: '%.50s'", pergunta)
        return QueryTransformada(
            query_original=pergunta,
            query_principal=pergunta,
            foi_transformada=False,
            motivo="gemini falhou",
        )

    query_reescrita = resultado.get("query_reescrita", "").strip()
    palavras_chave  = resultado.get("palavras_chave", [])

    # Valida: query reescrita deve ser mais rica que a original
    if not query_reescrita or len(query_reescrita) < len(pergunta) * 0.5:
        query_reescrita = pergunta

    # Trunca se muito longa
    query_reescrita = query_reescrita[:_MAX_QUERY_CHARS]

    logger.info(
        "🔄 Query transform: '%s' → '%s'",
        pergunta[:40], query_reescrita[:60],
    )

    return QueryTransformada(
        query_original=pergunta,
        query_principal=query_reescrita,
        palavras_chave=palavras_chave,
        foi_transformada=True,
        motivo="gemini rewrite",
    )


def _transformar_com_sub_queries(pergunta: str, fatos_str: str) -> QueryTransformada:
    """
    Sub-Query Decomposition para perguntas complexas com múltiplas intenções.
    """
    _PROMPT_SUB = """Decompõe a pergunta complexa abaixo em sub-perguntas simples.

Fatos do aluno:
{fatos}

Pergunta: {pergunta}

Cria 2-3 sub-perguntas técnicas que juntas respondem à pergunta original.
Responda APENAS com JSON:
{{"query_principal": "query abrangente",
  "sub_queries": ["sub-query 1", "sub-query 2", "sub-query 3"],
  "palavras_chave": ["termo1", "termo2"]}}
"""

    prompt = _PROMPT_SUB.format(fatos=fatos_str, pergunta=pergunta)

    resultado = chamar_gemini_estruturado(
        prompt=prompt,
        schema_descricao='{"query_principal": "str", "sub_queries": ["str"], "palavras_chave": ["str"]}',
        temperatura=0.1,
    )

    if not resultado:
        # Fallback para transformação simples
        return _transformar_query_simples(pergunta, fatos_str)

    query_principal = resultado.get("query_principal", pergunta).strip()[:_MAX_QUERY_CHARS]
    sub_queries = [q.strip()[:_MAX_QUERY_CHARS] for q in resultado.get("sub_queries", []) if q.strip()]
    palavras_chave = resultado.get("palavras_chave", [])

    logger.info(
        "🔄 Sub-queries: '%s' → principal='%s' + %d sub-queries",
        pergunta[:40], query_principal[:50], len(sub_queries),
    )

    return QueryTransformada(
        query_original=pergunta,
        query_principal=query_principal,
        sub_queries=sub_queries[:3],   # Máximo 3 sub-queries
        palavras_chave=palavras_chave,
        foi_transformada=True,
        motivo="sub-query decomposition",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Heurísticas de decisão
# ─────────────────────────────────────────────────────────────────────────────

def _precisa_transformar(pergunta: str, fatos: list[Fato] | None) -> bool:
    """
    Decide se a query precisa de transformação.

    NÃO transforma se:
    - É muito curta (menu ou saudação)
    - Já contém termos técnicos do domínio UEMA
    - Não há fatos do utilizador (sem contexto = sem benefício de reescrita)

    TRANSFORMA se:
    - É uma pergunta vaga/coloquial
    - Há fatos relevantes que podem enriquecer a busca
    """
    if len(pergunta) < _MIN_CHARS_PARA_TRANSFORM:
        return False

    # Verifica presença de termos técnicos
    pergunta_lower = _normalizar(pergunta)
    termos_encontrados = sum(1 for t in _TERMOS_JA_TECNICOS if t in pergunta_lower)

    # Se tem 2+ termos técnicos → já é boa o suficiente
    if termos_encontrados >= 2:
        return False

    # Se não tem fatos e a query já tem pelo menos 1 termo técnico → skip
    if termos_encontrados >= 1 and not fatos:
        return False

    return True


def _e_pergunta_complexa(pergunta: str) -> bool:
    """
    Detecta perguntas que contêm múltiplas intenções distintas.
    Candidatas para Sub-Query Decomposition.
    """
    # Indicadores de múltiplas intenções
    conectores = ["e", "também", "além disso", "e também", "e qual", "e quando", "e como"]
    pergunta_lower = pergunta.lower()

    # Pergunta longa + conectores = provavelmente complexa
    if len(pergunta) > 80 and any(c in pergunta_lower for c in conectores):
        return True

    # Múltiplos pontos de interrogação (raramente mas acontece)
    if pergunta.count("?") > 1:
        return True

    return False


def _normalizar(texto: str) -> str:
    """Normaliza texto removendo acentos para comparação de termos técnicos."""
    import unicodedata
    s = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode()
    return s.lower()