"""
domain/router.py — Roteamento por intenção (stateless)
=======================================================
SEM Redis. SEM I/O. Puro regex.
Testável com um simples assert direto:
    assert analisar("data de matrícula", EstadoMenu.MAIN) == Rota.CALENDARIO
"""
from __future__ import annotations
import re
import unicodedata
from src.domain.entities import Rota, EstadoMenu

# =============================================================================
# Padrões de roteamento
# EDITAL avaliado antes de CALENDARIO para evitar ambiguidade:
# "data de inscrição do PAES" → EDITAL (não CALENDARIO)
# =============================================================================

_PADROES: list[tuple[Rota, re.Pattern]] = [
    (Rota.EDITAL, re.compile(
        r"paes|vestibular|processo.seletivo|inscri[çc][aã]o|edital|"
        r"vaga|vagas|cota|cotas|\bac\b|pcd|br.ppi|br.q|br.dc|ir.ppi|cfo|"
        r"rede.p[uú]blica|quilombola|ind[ií]gena|defici[eê]ncia|"
        r"ampla.concorr[eê]ncia|reserva.de.vaga|heteroidentifica|"
        r"classifica[çc][aã]o|convoca[çc][aã]o|resultado.final|chamada",
        re.IGNORECASE,
    )),
    (Rota.CALENDARIO, re.compile(
        r"calend[aá]rio|matr[ií]cula|rematr[ií]cula|semestre|"
        r"per[ií]odo.letivo|in[ií]cio.das.aulas|fim.das.aulas|"
        r"feriado|recesso|prova|avalia[çc][aã]o|substitutiva|"
        r"trancamento|banca|defesa|prazo|reingresso|retardatário|"
        r"2026\.1|2026\.2|letivo",
        re.IGNORECASE,
    )),
    (Rota.CONTATOS, re.compile(
        r"contato|e.?mail|telefone|ramal|ligar|falar.com|"
        r"coordena[çc][aã]o|coordenador|diretor|secretaria|"
        r"\bprog\b|proexae|prppg|prad|\bctic\b|\bti\b|suporte|"
        r"cecen|cesb|cesc|ccsa|ceea|reitoria|vice.reitor",
        re.IGNORECASE,
    )),
]

# Mapa de submenu ativo → rota forçada
_ROTA_POR_ESTADO: dict[EstadoMenu, Rota] = {
    EstadoMenu.SUB_CALENDARIO: Rota.CALENDARIO,
    EstadoMenu.SUB_EDITAL:     Rota.EDITAL,
    EstadoMenu.SUB_CONTATOS:   Rota.CONTATOS,
}


def _norm(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode()
    return s.lower().strip()


def analisar(texto: str, estado: EstadoMenu = EstadoMenu.MAIN) -> Rota:
    """
    Determina a Rota (intenção) de uma mensagem.
    Pura, sem I/O, testável com assert direto.
    """
    # 1. Rota forçada pelo submenu ativo
    if estado in _ROTA_POR_ESTADO:
        return _ROTA_POR_ESTADO[estado]

    # 2. Detecção por padrão (EDITAL primeiro)
    txt = _norm(texto)
    for rota, pattern in _PADROES:
        if pattern.search(txt):
            return rota

    return Rota.GERAL