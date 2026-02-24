"""
tests/unit/test_validator.py — Testes de agent/validator.py
============================================================
Sem I/O. Puro Python.
Execute: pytest tests/unit/test_validator.py -v
"""
import pytest
from src.agent.validator import validar
from src.agent.state import AgentState
from src.domain.entities import Rota


def _state():
    return AgentState(
        user_id="test_user",
        session_id="test_session",
        mensagem_original="qual a data de matrícula?",
        rota=Rota.CALENDARIO,
    )


class TestValidador:
    def test_output_valido_passa(self):
        result = validar(_state(), "A matrícula de veteranos ocorre de 03/02 a 07/02/2026.")
        assert result.valido is True
        assert "matrícula" in result.output.lower()

    def test_output_vazio_invalida(self):
        result = validar(_state(), "")
        assert result.valido is False

    def test_output_none_invalida(self):
        result = validar(_state(), None)
        assert result.valido is False

    def test_string_langchain_max_iterations_invalida(self):
        result = validar(_state(), "Agent stopped due to max iterations.")
        assert result.valido is False
        assert "Agent stopped" not in result.output

    def test_string_langchain_timeout_invalida(self):
        result = validar(_state(), "Agent stopped due to iteration limit or time limit.")
        assert result.valido is False

    def test_string_parsing_error_invalida(self):
        result = validar(_state(), "parsing error")
        assert result.valido is False

    def test_output_muito_curto_invalida(self):
        result = validar(_state(), "Ok.")
        assert result.valido is False

    def test_output_com_whitespace_e_valido(self):
        result = validar(_state(), "  A data é 10 de março de 2026.  ")
        assert result.valido is True
        assert result.output == "A data é 10 de março de 2026."