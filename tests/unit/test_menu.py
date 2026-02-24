"""
tests/unit/test_menu.py — Testes de domain/menu.py
===================================================
Sem Redis. Sem Groq. Sem Docker. Puro Python.
Execute: pytest tests/unit/test_menu.py -v
"""
import pytest
from src.domain.menu import processar_mensagem, MENU_PRINCIPAL, TEXTO_SUBMENU
from src.domain.entities import EstadoMenu


class TestMenuPrincipal:
    def test_saudacao_oi_retorna_menu_principal(self):
        r = processar_mensagem("oi", EstadoMenu.MAIN)
        assert r["type"] == "menu_principal"
        assert r["novo_estado"] == EstadoMenu.MAIN
        assert r["content"] == MENU_PRINCIPAL

    def test_saudacao_ola_retorna_menu_principal(self):
        r = processar_mensagem("olá", EstadoMenu.MAIN)
        assert r["type"] == "menu_principal"

    def test_opcao_1_vai_para_calendario(self):
        r = processar_mensagem("1", EstadoMenu.MAIN)
        assert r["type"] == "submenu"
        assert r["novo_estado"] == EstadoMenu.SUB_CALENDARIO
        assert r["content"] == TEXTO_SUBMENU[EstadoMenu.SUB_CALENDARIO]

    def test_opcao_2_vai_para_edital(self):
        r = processar_mensagem("2", EstadoMenu.MAIN)
        assert r["type"] == "submenu"
        assert r["novo_estado"] == EstadoMenu.SUB_EDITAL

    def test_opcao_3_vai_para_contatos(self):
        r = processar_mensagem("3", EstadoMenu.MAIN)
        assert r["type"] == "submenu"
        assert r["novo_estado"] == EstadoMenu.SUB_CONTATOS

    def test_alias_calendario_vai_para_submenu(self):
        r = processar_mensagem("calendário", EstadoMenu.MAIN)
        assert r["type"] == "submenu"
        assert r["novo_estado"] == EstadoMenu.SUB_CALENDARIO

    def test_alias_edital_vai_para_submenu(self):
        r = processar_mensagem("edital", EstadoMenu.MAIN)
        assert r["type"] == "submenu"
        assert r["novo_estado"] == EstadoMenu.SUB_EDITAL

    def test_texto_livre_vai_para_llm(self):
        r = processar_mensagem("quando é a matrícula?", EstadoMenu.MAIN)
        assert r["type"] == "llm"
        assert r["prompt"] == "quando é a matrícula?"
        assert r["content"] is None


class TestSubmenus:
    def test_opcao_numerica_submenu_calendario_expande_prompt(self):
        r = processar_mensagem("1", EstadoMenu.SUB_CALENDARIO)
        assert r["type"] == "llm"
        assert "matrícula" in r["prompt"].lower() or "matricula" in r["prompt"].lower()
        assert r["novo_estado"] == EstadoMenu.MAIN

    def test_opcao_numerica_submenu_edital_expande_prompt(self):
        r = processar_mensagem("2", EstadoMenu.SUB_EDITAL)
        assert r["type"] == "llm"
        assert "documento" in r["prompt"].lower() or "inscrever" in r["prompt"].lower()

    def test_voltar_de_submenu_retorna_main(self):
        r = processar_mensagem("voltar", EstadoMenu.SUB_CALENDARIO)
        assert r["type"] == "menu_principal"
        assert r["novo_estado"] == EstadoMenu.MAIN

    def test_texto_livre_em_submenu_vai_para_llm(self):
        r = processar_mensagem("preciso de ajuda com as datas", EstadoMenu.SUB_CALENDARIO)
        assert r["type"] == "llm"
        assert r["prompt"] == "preciso de ajuda com as datas"

    def test_opcao_invalida_em_submenu_vai_para_llm(self):
        r = processar_mensagem("9", EstadoMenu.SUB_EDITAL)
        assert r["type"] == "llm"