"""
tests/unit/test_router.py — Testes de domain/router.py
=======================================================
Sem Redis. Sem Groq. Puro Python.
Execute: pytest tests/unit/test_router.py -v
"""
import pytest
from src.domain.router import analisar
from src.domain.entities import Rota, EstadoMenu


class TestRotaCalendario:
    def test_matricula(self):
        assert analisar("quando é a matrícula") == Rota.CALENDARIO

    def test_rematricula(self):
        assert analisar("prazo de rematrícula 2026") == Rota.CALENDARIO

    def test_semestre(self):
        assert analisar("início do semestre letivo") == Rota.CALENDARIO

    def test_prova(self):
        assert analisar("data da prova final") == Rota.CALENDARIO

    def test_trancamento(self):
        assert analisar("como trancar matrícula") == Rota.CALENDARIO

    def test_feriado(self):
        assert analisar("quais os feriados de março") == Rota.CALENDARIO


class TestRotaEdital:
    def test_paes(self):
        assert analisar("processo seletivo PAES 2026") == Rota.EDITAL

    def test_inscricao_paes_nao_confunde_com_calendario(self):
        # "inscrição" + "data" → deve ser EDITAL, não CALENDARIO
        assert analisar("data de inscrição do PAES") == Rota.EDITAL

    def test_vagas(self):
        assert analisar("quantas vagas para engenharia") == Rota.EDITAL

    def test_cotas(self):
        assert analisar("como funcionam as cotas BR-PPI") == Rota.EDITAL

    def test_documentos(self):
        assert analisar("quais documentos para inscrição") == Rota.EDITAL


class TestRotaContatos:
    def test_email(self):
        assert analisar("qual o email da PROG") == Rota.CONTATOS

    def test_telefone(self):
        assert analisar("telefone da secretaria") == Rota.CONTATOS

    def test_ctic(self):
        assert analisar("contato do CTIC") == Rota.CONTATOS

    def test_coordenacao(self):
        assert analisar("coordenação do curso de direito") == Rota.CONTATOS


class TestRotaGeral:
    def test_texto_generico(self):
        assert analisar("oi tudo bem") == Rota.GERAL

    def test_texto_curto_sem_palavras_chave(self):
        assert analisar("ok") == Rota.GERAL


class TestRotaForcadaPorEstado:
    def test_submenu_calendario_forca_rota(self):
        # Mesmo texto genérico → CALENDARIO porque está no submenu
        assert analisar("me ajuda", EstadoMenu.SUB_CALENDARIO) == Rota.CALENDARIO

    def test_submenu_edital_forca_rota(self):
        assert analisar("me ajuda", EstadoMenu.SUB_EDITAL) == Rota.EDITAL

    def test_submenu_contatos_forca_rota(self):
        assert analisar("me ajuda", EstadoMenu.SUB_CONTATOS) == Rota.CONTATOS