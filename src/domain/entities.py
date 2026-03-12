"""
domain/entities.py — Entidades de domínio puras (v4)
======================================================
Sem Redis. Sem I/O. Tipos que trafegam entre todas as camadas.

MUDANÇAS v4:
  Rota.WIKI adicionada → tool_wiki_ctic.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Rota(str, Enum):
    CALENDARIO = "CALENDARIO"
    EDITAL     = "EDITAL"
    CONTATOS   = "CONTATOS"
    WIKI       = "WIKI"       # ← NOVO v4 — Wiki CTIC/UEMA
    GERAL      = "GERAL"


class EstadoMenu(str, Enum):
    MAIN           = "MAIN"
    SUB_CALENDARIO = "SUB_CALENDARIO"
    SUB_EDITAL     = "SUB_EDITAL"
    SUB_CONTATOS   = "SUB_CONTATOS"


@dataclass
class Mensagem:
    user_id:   str
    chat_id:   str
    body:      str
    timestamp: datetime = field(default_factory=datetime.now)
    has_media: bool     = False
    msg_type:  str      = "text"


@dataclass
class RAGResult:
    conteudo: str
    source:   str
    score:    float = 0.0

    @property
    def encontrou(self) -> bool:
        return bool(self.conteudo.strip())


@dataclass
class AgentResponse:
    conteudo:       str
    rota:           Rota            = Rota.GERAL
    tokens_entrada: int             = 0
    tokens_saida:   int             = 0
    latencia_ms:    int             = 0
    iteracoes:      int             = 0
    sucesso:        bool            = True
    rag_results:    list[RAGResult] = field(default_factory=list)

    @property
    def tokens_total(self) -> int:
        return self.tokens_entrada + self.tokens_saida