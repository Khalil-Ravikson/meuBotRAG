"""
rag/calendar_parser.py — Parser de Eventos do Calendário Acadêmico UEMA
========================================================================

PROBLEMA QUE RESOLVE:
─────────────────────
  Os chunks do calendário no Redis têm este formato textual:
    "EVENTO: Matrícula de veteranos | DATA: 03/02/2026 a 07/02/2026 | SEM: 2026.1"
    "EVENTO: Início das aulas | DATA: 10/02/2026 | SEM: 2026.1"
    "EVENTO: Feriado Carnaval | DATA: 02/03/2026 e 03/03/2026 | SEM: 2026.1"

  Para notificar os alunos, precisamos:
    1. Buscar chunks que mencionem datas próximas
    2. Parsear o texto e extrair data de início
    3. Calcular quantos dias faltam
    4. Montar mensagem personalizada

ESTRATÉGIA (sem LLM, sem custo de tokens):
───────────────────────────────────────────
  1. Busca no Redis por BM25 usando a data de hoje + amanhã como query
     Ex: query="03/02/2026 02/02/2026 04/02/2026 prazo matrícula"
  
  2. Regex para extrair datas dos chunks retornados:
     Padrões: "DD/MM/YYYY", "DD/MM a DD/MM/YYYY", "DD de MÊS de YYYY"
  
  3. Filtra apenas eventos com início nos próximos N dias

  VANTAGEM: zero tokens Gemini — tudo local no Redis + regex Python.
  CUSTO: ~3ms por verificação (BM25 no Redis).

TIPOS DE EVENTO RECONHECIDOS:
  🎓 Matrícula/Rematrícula → aviso 3 dias antes, lembrete no dia
  📝 Provas/Avaliações → aviso 3 dias antes
  📅 Trancamento → aviso 5 dias antes (prazo crítico)
  🏫 Início de semestre → aviso 1 dia antes
  🎉 Feriados/Recessos → informativo (sem urgência)
"""
from __future__ import annotations

import logging
import re
import struct
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Meses em português → número
# ─────────────────────────────────────────────────────────────────────────────

_MESES_PT = {
    "janeiro": 1,  "fevereiro": 2,  "março": 3,    "marco": 3,
    "abril": 4,    "maio": 5,       "junho": 6,
    "julho": 7,    "agosto": 8,     "setembro": 9,
    "outubro": 10, "novembro": 11,  "dezembro": 12,
}

# ─────────────────────────────────────────────────────────────────────────────
# Padrões de data (compilados uma vez)
# ─────────────────────────────────────────────────────────────────────────────

# DD/MM/YYYY  ou  DD/MM/YY
_RE_DATA_BARRA  = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")

# "03 de fevereiro de 2026"
_RE_DATA_EXTENSO = re.compile(
    r"\b(\d{1,2})\s+de\s+(" + "|".join(_MESES_PT) + r")\s+de\s+(\d{4})\b",
    re.IGNORECASE,
)

# Padrão "EVENTO: ... | DATA: ... | SEM: ..."
_RE_EVENTO = re.compile(
    r"EVENTO:\s*(?P<nome>[^|]+)\|?\s*"
    r"DATA:\s*(?P<data>[^|]+)\|?\s*"
    r"(?:SEM:\s*(?P<semestre>[^\n|]+))?",
    re.IGNORECASE,
)

# Categoria do evento por palavras-chave
_CATEGORIAS = {
    "urgente": [
        "matrícula", "rematrícula", "trancamento", "reingresso",
        "inscrição", "inscricao", "prazo", "retardatário",
    ],
    "prova": [
        "prova", "avaliação", "avaliacao", "substitutiva",
        "banca", "defesa", "exame",
    ],
    "inicio": [
        "início das aulas", "inicio das aulas", "começo",
        "abertura", "retorno",
    ],
    "feriado": [
        "feriado", "recesso", "ponto facultativo",
    ],
    "administrativo": [
        "lançamento de notas", "lancamento", "colação",
        "formatura", "diploma",
    ],
}

# Dias de antecedência para notificar por categoria
_ANTECEDENCIA_DIAS = {
    "urgente":        [5, 3, 1, 0],   # notifica em T-5, T-3, T-1 e no dia
    "prova":          [3, 1, 0],
    "inicio":         [1, 0],
    "feriado":        [1],             # só avisa 1 dia antes
    "administrativo": [3, 1],
}

# Emojis por categoria
_EMOJI_CATEGORIA = {
    "urgente":        "⚠️",
    "prova":          "📝",
    "inicio":         "🏫",
    "feriado":        "🎉",
    "administrativo": "📋",
    "outro":          "📅",
}

# ─────────────────────────────────────────────────────────────────────────────
# Tipos de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventoCalendario:
    """Evento extraído e parseado do calendário acadêmico."""
    nome:       str
    data_inicio: date
    data_fim:   date | None   = None
    semestre:   str           = ""
    categoria:  str           = "outro"
    chunk_raw:  str           = ""        # texto original do chunk

    @property
    def dias_restantes(self) -> int:
        """Dias até o evento a partir de hoje."""
        return (self.data_inicio - date.today()).days

    @property
    def emoji(self) -> str:
        return _EMOJI_CATEGORIA.get(self.categoria, "📅")

    @property
    def deve_notificar_hoje(self) -> bool:
        """Verifica se hoje é um dos dias de notificação para este evento."""
        antecedencias = _ANTECEDENCIA_DIAS.get(self.categoria, [3, 1])
        return self.dias_restantes in antecedencias

    def mensagem_notificacao(self, nome_usuario: str = "") -> str:
        """
        Gera a mensagem WhatsApp para notificar o aluno.
        Personalizada se nome_usuario for fornecido.
        """
        saudacao = f"Olá, {nome_usuario}! " if nome_usuario else "Olá! "

        dias = self.dias_restantes
        if dias == 0:
            urgencia = "🚨 *Hoje é o último dia!*"
        elif dias == 1:
            urgencia = "⏰ *Amanhã é o prazo!*"
        elif dias <= 3:
            urgencia = f"⏳ *Faltam {dias} dias!*"
        else:
            urgencia = f"📆 *Em {dias} dias:*"

        data_str = self.data_inicio.strftime("%d/%m/%Y")
        if self.data_fim and self.data_fim != self.data_inicio:
            data_str += f" a {self.data_fim.strftime('%d/%m/%Y')}"

        sem_str = f" (semestre {self.semestre})" if self.semestre else ""

        return (
            f"{saudacao}Lembrete do *Oráculo UEMA* 🎓\n\n"
            f"{urgencia}\n"
            f"{self.emoji} *{self.nome.strip()}*\n"
            f"📅 Data: *{data_str}*{sem_str}\n\n"
            f"Precisa de mais informações? É só me perguntar!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Funções principais
# ─────────────────────────────────────────────────────────────────────────────

def buscar_eventos_proximos(
    dias_frente: int = 7,
    dias_atras:  int = 0,
) -> list[EventoCalendario]:
    """
    Busca no Redis os eventos do calendário nos próximos N dias.

    COMO FUNCIONA:
      1. Gera queries com as datas do período (ex: "02/02 03/02 04/02")
      2. Faz busca híbrida (BM25 captura datas exatas muito bem)
      3. Parseia os chunks retornados com regex
      4. Filtra pelos eventos realmente dentro do período

    Parâmetros:
      dias_frente: quantos dias à frente olhar (padrão: 7)
      dias_atras:  quantos dias atrás incluir (padrão: 0 = só futuro)

    Retorna lista ordenada por data_inicio.
    """
    from src.infrastructure.redis_client import busca_hibrida
    from src.rag.embeddings import get_embeddings

    hoje        = date.today()
    data_inicio = hoje - timedelta(days=dias_atras)
    data_fim    = hoje + timedelta(days=dias_frente)

    # Monta query com as datas do período (BM25 é ótimo para datas exatas)
    datas_query = _gerar_query_datas(data_inicio, data_fim)
    logger.info(
        "🔍 Buscando eventos de %s a %s | query: '%s'",
        data_inicio, data_fim, datas_query[:60],
    )

    # Embedding da query
    try:
        embeddings = get_embeddings()
        vetor = embeddings.embed_query(f"prazo matrícula prova avaliação {datas_query}")
    except Exception as e:
        logger.error("❌ Falha no embedding do calendar_parser: %s", e)
        return []

    # Busca híbrida — k_text alto pois datas são keywords exatas
    try:
        resultados = busca_hibrida(
            query_text     = datas_query,
            query_embedding= vetor,
            source_filter  = "calendario-academico-2026.pdf",
            k_vector       = 6,
            k_text         = 12,   # BM25 é mais importante para datas
        )
    except Exception as e:
        logger.error("❌ Falha na busca do calendário: %s", e)
        return []

    # Parseia todos os chunks retornados
    eventos_vistos: set[str] = set()
    eventos: list[EventoCalendario] = []

    for chunk in resultados:
        conteudo = chunk.get("content", "")
        for evento in _parsear_chunk(conteudo):
            # Filtra pelo período
            if not (data_inicio <= evento.data_inicio <= data_fim):
                continue
            # Deduplicação por nome + data
            chave = f"{evento.nome.lower().strip()}:{evento.data_inicio}"
            if chave in eventos_vistos:
                continue
            eventos_vistos.add(chave)
            eventos.append(evento)

    eventos.sort(key=lambda e: e.data_inicio)
    logger.info("📅 Eventos encontrados no período: %d", len(eventos))
    return eventos


def buscar_eventos_para_notificar_hoje() -> list[EventoCalendario]:
    """
    Retorna apenas os eventos que devem ser notificados HOJE.
    Chamado pelo Celery Beat toda manhã às 8h.

    Lógica:
      - Busca eventos dos próximos 7 dias
      - Filtra os que têm `deve_notificar_hoje == True`
      - Ou seja: eventos em T-5, T-3, T-1 e T-0 conforme a categoria
    """
    todos = buscar_eventos_proximos(dias_frente=7)
    notificar = [e for e in todos if e.deve_notificar_hoje]
    logger.info(
        "🔔 Eventos para notificar hoje (%s): %d de %d encontrados",
        date.today(), len(notificar), len(todos),
    )
    return notificar


# ─────────────────────────────────────────────────────────────────────────────
# Parsers internos
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_chunk(texto: str) -> Iterator[EventoCalendario]:
    """
    Extrai eventos de um chunk de texto do calendário.
    Tenta primeiro o formato estruturado EVENTO:|DATA:, depois busca livre.
    """
    # Formato estruturado (resultado do LlamaParse/chunking hierárquico)
    for m in _RE_EVENTO.finditer(texto):
        nome      = m.group("nome").strip()
        data_raw  = m.group("data").strip()
        semestre  = (m.group("semestre") or "").strip()

        if not nome or not data_raw:
            continue

        datas = _extrair_datas(data_raw)
        if not datas:
            continue

        categoria = _classificar_evento(nome)
        yield EventoCalendario(
            nome        = nome,
            data_inicio = datas[0],
            data_fim    = datas[-1] if len(datas) > 1 else None,
            semestre    = semestre,
            categoria   = categoria,
            chunk_raw   = texto[:200],
        )

    # Fallback: busca livre por linhas com padrão "DATA: ..."
    # Útil para chunks de formatação irregular
    for linha in texto.splitlines():
        if "DATA:" not in linha.upper():
            continue
        datas = _extrair_datas(linha)
        if not datas:
            continue
        # Extrai o nome da parte antes do "DATA:"
        partes = re.split(r"DATA:", linha, flags=re.IGNORECASE, maxsplit=1)
        if not partes:
            continue
        nome_raw = partes[0].replace("EVENTO:", "").replace("|", "").strip()
        if len(nome_raw) < 5:
            continue

        categoria = _classificar_evento(nome_raw)
        yield EventoCalendario(
            nome        = nome_raw,
            data_inicio = datas[0],
            data_fim    = datas[-1] if len(datas) > 1 else None,
            categoria   = categoria,
            chunk_raw   = linha[:200],
        )


def _extrair_datas(texto: str) -> list[date]:
    """
    Extrai todas as datas de uma string.
    Suporta formatos DD/MM/YYYY e "DD de mês de YYYY".
    """
    datas: list[date] = []

    for m in _RE_DATA_BARRA.finditer(texto):
        try:
            d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            datas.append(d)
        except ValueError:
            pass

    for m in _RE_DATA_EXTENSO.finditer(texto):
        try:
            mes = _MESES_PT.get(m.group(2).lower())
            if mes:
                d = date(int(m.group(3)), mes, int(m.group(1)))
                datas.append(d)
        except ValueError:
            pass

    # Remove duplicatas e ordena
    return sorted(set(datas))


def _classificar_evento(nome: str) -> str:
    """Classifica o evento por palavras-chave no nome."""
    nome_lower = nome.lower()
    for categoria, keywords in _CATEGORIAS.items():
        if any(kw in nome_lower for kw in keywords):
            return categoria
    return "outro"


def _gerar_query_datas(inicio: date, fim: date) -> str:
    """
    Gera string de query BM25 com as datas do período.
    Ex: "02/02 03/02 04/02 05/02 prazo matrícula avaliação"
    """
    datas = []
    atual = inicio
    while atual <= fim:
        # Adiciona DD/MM e DD/MM/YYYY para maximizar hits no BM25
        datas.append(atual.strftime("%d/%m"))
        datas.append(atual.strftime("%d/%m/%Y"))
        atual += timedelta(days=1)

    termos_academicos = "matrícula prova avaliação prazo trancamento início feriado"
    return " ".join(datas[:14]) + " " + termos_academicos  # limita para não explodir a query


# ─────────────────────────────────────────────────────────────────────────────
# Utilitário de diagnóstico
# ─────────────────────────────────────────────────────────────────────────────

def listar_todos_eventos(limite: int = 50) -> list[EventoCalendario]:
    """
    Lista todos os eventos do calendário 2026 encontrados no Redis.
    Útil para debug: python -c "from src.rag.calendar_parser import listar_todos_eventos; ..."
    """
    return buscar_eventos_proximos(dias_frente=400, dias_atras=30)[:limite]