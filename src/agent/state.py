"""
agent/state.py — Estado do agente multi-step
============================================
Objeto de trabalho que passa por planner → executor → validator.
Imutável por convenção: modifique com os métodos abaixo, não diretamente.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from src.domain.entities import Rota, EstadoMenu, RAGResult


@dataclass
class AgentState:
    # ── Identificação ────────────────────────────────────────────────────────
    user_id:            str
    session_id:         str
    mensagem_original:  str
    chat_id:            str = ""

    # ── Roteamento ───────────────────────────────────────────────────────────
    rota:       Rota       = Rota.GERAL
    modo_menu:  EstadoMenu = EstadoMenu.MAIN

    # ── Prompt montado pelo router ────────────────────────────────────────────
    prompt_enriquecido: str = ""

    # ── Controle do loop ─────────────────────────────────────────────────────
    iteracao_atual: int = 0
    max_iteracoes:  int = 3

    # ── Plano e execução ─────────────────────────────────────────────────────
    plano:            list[str]  = field(default_factory=list)
    steps_executados: list[dict] = field(default_factory=list)

    # ── Resultados ───────────────────────────────────────────────────────────
    rag_results:    list[RAGResult] = field(default_factory=list)
    resposta_final: str | None      = None
    erro:           str | None      = None

    # ── Métricas ─────────────────────────────────────────────────────────────
    tokens_entrada: int      = 0
    tokens_saida:   int      = 0
    iniciado_em:    datetime = field(default_factory=datetime.now)

    # ── Contexto do usuário (vem de memory/) ─────────────────────────────────
    contexto_usuario: dict = field(default_factory=dict)

    # =========================================================================
    # Propriedades derivadas
    # =========================================================================

    @property
    def concluido(self) -> bool:
        return self.resposta_final is not None or self.erro is not None

    @property
    def atingiu_limite(self) -> bool:
        return self.iteracao_atual >= self.max_iteracoes

    @property
    def tokens_total(self) -> int:
        return self.tokens_entrada + self.tokens_saida

    @property
    def latencia_ms(self) -> int:
        return int((datetime.now() - self.iniciado_em).total_seconds() * 1000)

    # =========================================================================
    # Mutadores explícitos (facilita rastreamento em testes)
    # =========================================================================

    def registrar_step(self, acao: str, resultado: str, sucesso: bool = True) -> None:
        self.steps_executados.append({
            "iteracao": self.iteracao_atual,
            "acao":     acao,
            "resultado": resultado[:500],
            "sucesso":  sucesso,
        })

    def incrementar(self) -> None:
        self.iteracao_atual += 1

    def finalizar(self, resposta: str) -> None:
        self.resposta_final = resposta

    def falhar(self, motivo: str) -> None:
        self.erro = motivo
        self.resposta_final = (
            "Desculpe, tive uma dificuldade técnica. Tente novamente."
        )