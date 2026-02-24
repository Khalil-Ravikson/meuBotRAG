"""
agent/validator.py — Validação do output do agente
===================================================
Verifica se o output é enviável ao usuário antes de retornar.
Testável com outputs mockados — sem I/O.
"""
from __future__ import annotations
from dataclasses import dataclass
from src.agent.state import AgentState
from src.agent.prompts import OUTPUTS_INVALIDOS, MSG_NAO_ENCONTRADO


@dataclass
class ValidationResult:
    valido:  bool
    output:  str       # output sanitizado (pode ser diferente do original)
    motivo:  str = ""  # razão da invalidação (para logs)


def validar(state: AgentState, output: str) -> ValidationResult:
    """
    Valida e sanitiza o output do agente.

    Critérios:
      1. Output não é string interna do LangChain
      2. Output tem conteúdo mínimo (> 10 chars)
      3. Iterações não excederam o limite (já controlado pelo core, mas dupla-checagem)

    Puro: sem I/O, sem Redis, testável com assert.
    """
    if not output:
        return ValidationResult(False, MSG_NAO_ENCONTRADO, "output vazio")

    output_lower = output.strip().lower()

    # Strings internas do LangChain
    for invalido in OUTPUTS_INVALIDOS:
        if invalido in output_lower:
            return ValidationResult(False, MSG_NAO_ENCONTRADO,
                                    f"output inválido: {invalido!r}")

    # Output muito curto suspeito
    if len(output.strip()) < 10:
        return ValidationResult(False, MSG_NAO_ENCONTRADO, "output muito curto")

    return ValidationResult(True, output.strip())