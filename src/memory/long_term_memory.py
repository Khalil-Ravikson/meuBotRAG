"""
memory/long_term_memory.py — Memória Factual de Longo Prazo
=============================================================

O QUE SÃO "FATOS" NESTE CONTEXTO?
────────────────────────────────────
  Fatos são fragmentos de conhecimento verificável sobre o utilizador,
  extraídos das suas conversas passadas de forma assíncrona.

  Exemplos de fatos reais para um aluno da UEMA:
    - "Aluno do curso de Engenharia Civil, turno noturno"
    - "Inscrito no PAES 2026 na categoria BR-PPI"
    - "Já fez matrícula veterano no semestre 2026.1"
    - "Dúvida recorrente sobre trancamento de matrícula"
    - "Solicita contato da coordenação de Engenharia Civil"

COMO OS FATOS ELIMINAM ALUCINAÇÕES:
─────────────────────────────────────
  SEM FATOS:
    Aluno: "onde fico minha prova?"
    Sistema: Não sabe qual curso → busca genericamente → retorna info vaga
    → Alucinação: inventa sala ou data

  COM FATOS:
    Long-Term Memory: "Aluno de Engenharia Civil, turno noturno"
    Query Transformer: "onde fico minha prova?" + fato
    → "local de prova Engenharia Civil turno noturno avaliação final 2026.1"
    Busca híbrida: encontra exatamente o chunk certo
    → Resposta precisa, sem alucinação

ARQUITETURA NO REDIS:
─────────────────────
  Dois tipos de armazenamento complementares:

  1. mem:facts:list:{user_id}
     → Lista de fatos como strings (rápida para leitura em ordem)
     → TTL: 30 dias
     → Usado quando queremos os N fatos mais recentes (Quick Recall)

  2. mem:facts:vec:{user_id}:{fact_hash}
     → JSON com texto + embedding (para busca semântica)
     → Permite encontrar fatos RELEVANTES para a pergunta atual
     → TTL: 30 dias

  QUANDO USAR CADA UM:
    buscar_fatos_recentes()   → usa a Lista (mais rápido, sem embedding)
    buscar_fatos_relevantes() → usa o índice vetorial (mais preciso)

EXTRAÇÃO DE FATOS (feita por memory/extractor.py):
────────────────────────────────────────────────────
  A extração NÃO acontece em tempo real (economiza tokens no caminho crítico).
  É feita de forma assíncrona:
    - Após cada resposta → trigger para extração em background
    - Usa o Gemini com temperatura 0.1 para ser conservador
    - Fatos são normalizados antes de guardar (evita duplicados)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass

from src.infrastructure.redis_client import (
    VECTOR_DIM,
    get_redis,
    get_redis_text,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

_TTL_FATOS       = 86400 * 30   # 30 dias — fatos persistem entre semestres
_MAX_FATOS_USER  = 50           # Limite por utilizador para não inflar RAM
_MAX_FATOS_QUERY = 5            # Máximo de fatos retornados por busca

_PREFIX_FATOS_LIST = "mem:facts:list:"  # Para Quick Recall (lista)
_PREFIX_FATOS_VEC  = "mem:facts:vec:"   # Para Semantic Recall (vetores)

# Threshold de similaridade para fatos relevantes
_THRESHOLD_FATO_RELEVANTE = 0.65


# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Fato:
    """Fato factual sobre um utilizador."""
    texto:     str
    user_id:   str
    timestamp: int = 0
    score:     float = 0.0   # Score de relevância (preenchido na busca)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = int(time.time())

    @property
    def hash_id(self) -> str:
        """Hash determinístico do texto — evita duplicados."""
        return hashlib.md5(self.texto.lower().strip().encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Guardar fatos
# ─────────────────────────────────────────────────────────────────────────────

def guardar_fato(user_id: str, texto_fato: str) -> bool:
    """
    Converte um facto em vetor e guarda no Redis (duas estruturas).

    FLUXO:
      1. Normaliza o texto (lowercase, strip)
      2. Verifica se fato já existe (por hash) → evita duplicados
      3. Computa embedding do fato (BAAI/bge-m3, CPU local)
      4. Guarda JSON com embedding no Redis (mem:facts:vec:...)
      5. Guarda texto na lista ordenada (mem:facts:list:...)
      6. Aplica ltrim para não ultrapassar _MAX_FATOS_USER

    POR QUE GUARDAR O EMBEDDING DO FATO?
      Quando o aluno pergunta algo, buscamos quais fatos são
      SEMANTICAMENTE RELEVANTES para aquela pergunta específica.

      Exemplo:
        Pergunta: "quando é o início das aulas?"
        Fatos do aluno: [
          "Aluno de Engenharia Civil" → similaridade: 0.45 (baixa)
          "Dúvida sobre início do 2026.1" → similaridade: 0.89 (alta!) ✓
          "Inscrito via BR-PPI" → similaridade: 0.32 (baixa)
        ]
        → Apenas o fato relevante entra no contexto

    Retorna True se guardado com sucesso, False se já existia ou erro.
    """
    texto_normalizado = texto_fato.strip()
    if not texto_normalizado or len(texto_normalizado) < 10:
        return False

    fato = Fato(texto=texto_normalizado, user_id=user_id)

    r       = get_redis()
    r_text  = get_redis_text()

    # ── Verifica duplicado ───────────────────────────────────────────────────
    key_vec = f"{_PREFIX_FATOS_VEC}{user_id}:{fato.hash_id}"
    if r.exists(key_vec):
        logger.debug("ℹ️  Fato já existe [%s]: %.60s", user_id, texto_normalizado)
        return False

    # ── Computa embedding ────────────────────────────────────────────────────
    try:
        from src.rag.vector_store import get_embeddings
        embeddings_model = get_embeddings()
        vetor = embeddings_model.embed_query(texto_normalizado)
    except Exception as e:
        logger.error("❌ Falha ao computar embedding do fato [%s]: %s", user_id, e)
        # Guarda sem embedding (ainda útil para Quick Recall)
        vetor = []

    # ── Guarda JSON + embedding no Redis ─────────────────────────────────────
    doc = {
        "texto":     texto_normalizado,
        "user_id":   user_id,
        "timestamp": fato.timestamp,
        "embedding": vetor,
    }
    try:
        r.json().set(key_vec, "$", doc)
        r.expire(key_vec, _TTL_FATOS)
    except Exception as e:
        logger.error("❌ Falha ao guardar vetor do fato: %s", e)
        return False

    # ── Guarda na lista (para Quick Recall) ──────────────────────────────────
    key_list = f"{_PREFIX_FATOS_LIST}{user_id}"
    try:
        r_text.lpush(key_list, texto_normalizado)
        r_text.ltrim(key_list, 0, _MAX_FATOS_USER - 1)
        r_text.expire(key_list, _TTL_FATOS)
    except Exception as e:
        logger.warning("⚠️  Falha ao guardar fato na lista: %s", e)

    logger.info("💾 Fato guardado [%s]: %.80s", user_id, texto_normalizado)
    return True


def guardar_fatos_batch(user_id: str, fatos: list[str]) -> int:
    """
    Guarda múltiplos fatos de uma vez (usado pela rotina de extração).
    Retorna o número de fatos novos guardados.
    """
    guardados = 0
    for texto in fatos:
        if guardar_fato(user_id, texto):
            guardados += 1
    if guardados:
        logger.info("💾 Batch: %d/%d fatos novos para [%s]", guardados, len(fatos), user_id)
    return guardados


# ─────────────────────────────────────────────────────────────────────────────
# Buscar fatos
# ─────────────────────────────────────────────────────────────────────────────

def buscar_fatos_relevantes(
    user_id: str,
    pergunta: str,
    max_fatos: int = _MAX_FATOS_QUERY,
) -> list[Fato]:
    """
    Busca fatos do utilizador semanticamente relevantes para a pergunta.

    ALGORITMO:
    ──────────
      1. Computa embedding da pergunta
      2. Carrega todos os fatos do utilizador (tipicamente < 50)
      3. Calcula similaridade coseno entre a pergunta e cada fato
      4. Retorna os top-N fatos acima do threshold

    POR QUE NÃO USAMOS O ÍNDICE REDIS PARA ISTO?
      O índice IDX_TOOLS usa FLAT search (exato) que é eficiente para
      poucos documentos. Para fatos de um utilizador específico,
      o número é ainda menor (< 50) e fazer scan + similaridade local
      é mais simples e igualmente rápido para este volume.

      Alternativa futura: criar um índice por utilizador com
      tag filter "@user_id:{...}" — implementável quando escalar.

    Parâmetros:
      user_id:   ID do utilizador (número de WhatsApp)
      pergunta:  Pergunta atual do utilizador
      max_fatos: Máximo de fatos a retornar

    Retorna: lista de Fatos ordenados por relevância decrescente
    """
    r = get_redis()

    # ── Carrega todos os fatos com embedding do utilizador ──────────────────
    fatos_raw = _carregar_fatos_com_embedding(r, user_id)
    if not fatos_raw:
        logger.debug("ℹ️  Sem fatos para [%s]", user_id)
        return []

    # ── Computa embedding da pergunta ────────────────────────────────────────
    try:
        from src.rag.vector_store import get_embeddings
        embeddings_model = get_embeddings()
        vetor_pergunta = embeddings_model.embed_query(pergunta)
    except Exception as e:
        logger.warning("⚠️  Falha embedding pergunta para fatos: %s", e)
        # Fallback: retorna fatos recentes sem filtragem semântica
        return buscar_fatos_recentes(user_id, max_fatos)

    # ── Calcula similaridade coseno ──────────────────────────────────────────
    fatos_com_score: list[Fato] = []
    for fato_dict in fatos_raw:
        vetor_fato = fato_dict.get("embedding", [])
        if not vetor_fato or len(vetor_fato) != len(vetor_pergunta):
            continue
        score = _similaridade_coseno(vetor_pergunta, vetor_fato)
        if score >= _THRESHOLD_FATO_RELEVANTE:
            fatos_com_score.append(Fato(
                texto=fato_dict.get("texto", ""),
                user_id=user_id,
                timestamp=fato_dict.get("timestamp", 0),
                score=score,
            ))

    # ── Ordena por score e retorna top-N ─────────────────────────────────────
    fatos_ordenados = sorted(fatos_com_score, key=lambda f: f.score, reverse=True)
    resultado = fatos_ordenados[:max_fatos]

    if resultado:
        logger.debug(
            "🧠 Fatos relevantes [%s]: %d/%d | top score=%.3f | '%s'",
            user_id, len(resultado), len(fatos_raw),
            resultado[0].score, resultado[0].texto[:50],
        )

    return resultado


def buscar_fatos_recentes(user_id: str, max_fatos: int = 5) -> list[Fato]:
    """
    Retorna os N fatos mais recentes do utilizador (sem semântica).
    Fallback rápido quando o embedding não está disponível.
    """
    r_text = get_redis_text()
    key = f"{_PREFIX_FATOS_LIST}{user_id}"
    try:
        textos = r_text.lrange(key, 0, max_fatos - 1)
        return [Fato(texto=t, user_id=user_id) for t in textos if t]
    except Exception:
        return []


def fatos_como_string(fatos: list[Fato]) -> str:
    """
    Formata lista de fatos como string para injeção no prompt.
    Mantém formato compacto para economizar tokens.
    """
    if not fatos:
        return ""
    return "\n".join(f"- {f.texto}" for f in fatos)


def listar_todos_fatos(user_id: str) -> list[str]:
    """
    Lista todos os fatos de um utilizador (para debug e endpoint /fatos).
    """
    r_text = get_redis_text()
    key = f"{_PREFIX_FATOS_LIST}{user_id}"
    try:
        return r_text.lrange(key, 0, _MAX_FATOS_USER - 1) or []
    except Exception:
        return []


def apagar_fatos(user_id: str) -> None:
    """Remove todos os fatos de um utilizador (GDPR / privacidade)."""
    r = get_redis()
    r_text = get_redis_text()

    # Remove lista
    r_text.delete(f"{_PREFIX_FATOS_LIST}{user_id}")

    # Remove chaves vetoriais com SCAN
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = r.scan(cursor, match=f"{_PREFIX_FATOS_VEC}{user_id}:*", count=100)
        if keys:
            r.delete(*keys)
            deleted += len(keys)
        if cursor == 0:
            break

    logger.info("🗑️  Fatos apagados para [%s]: lista + %d vetores", user_id, deleted)


# ─────────────────────────────────────────────────────────────────────────────
# Funções internas
# ─────────────────────────────────────────────────────────────────────────────

def _carregar_fatos_com_embedding(r, user_id: str) -> list[dict]:
    """
    Carrega todos os JSONs de fatos do utilizador do Redis.
    Usa SCAN para não bloquear o Redis em produção.
    """
    pattern = f"{_PREFIX_FATOS_VEC}{user_id}:*"
    fatos = []
    cursor = 0

    while True:
        cursor, keys = r.scan(cursor, match=pattern, count=100)
        for key in keys:
            try:
                doc = r.json().get(key, "$")
                if doc:
                    item = doc[0] if isinstance(doc, list) else doc
                    fatos.append(item)
            except Exception:
                pass
        if cursor == 0:
            break

    return fatos


def _similaridade_coseno(v1: list[float], v2: list[float]) -> float:
    """
    Calcula similaridade coseno entre dois vetores.
    Implementação pura Python para não depender de numpy no caminho crítico.

    Para vetores BAAI/bge-m3 (já normalizados), a similaridade coseno
    é simplesmente o produto escalar.

    NOTA: embeddings_model.embed_query() com normalize_embeddings=True
    já retorna vetores unitários, então o produto escalar = coseno.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    # Produto escalar (suficiente para vetores normalizados)
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, dot))   # Clamp [0, 1]