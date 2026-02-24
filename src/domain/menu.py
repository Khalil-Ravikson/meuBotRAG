"""
domain/menu.py — Lógica de menu pura (stateless)
=================================================
SEM Redis. SEM I/O. Apenas regras de negócio.
O estado do usuário vem de fora — injetado por application/handle_message.py.

Testável sem nenhum mock:
    resultado = processar_mensagem("oi", EstadoMenu.MAIN)
    assert resultado["type"] == "menu_principal"
"""
from __future__ import annotations
import re
import unicodedata
from src.domain.entities import EstadoMenu

# =============================================================================
# Textos dos menus (única fonte da verdade)
# =============================================================================

MENU_PRINCIPAL = (
    "👋 *Olá! Sou o Assistente Virtual da UEMA.*\n\n"
    "Escolha uma opção:\n\n"
    "📅 *1.* Calendário Acadêmico\n"
    "📋 *2.* Edital PAES 2026\n"
    "📞 *3.* Contatos e E-mails\n\n"
    "_Ou digite sua dúvida diretamente._"
)

TEXTO_SUBMENU: dict[EstadoMenu, str] = {
    EstadoMenu.SUB_CALENDARIO: (
        "📅 *Calendário Acadêmico 2026*\n\n"
        "*1.* Matrícula e Rematrícula\n"
        "*2.* Início e Fim de Semestre\n"
        "*3.* Feriados e Recessos\n"
        "*4.* Provas e Avaliações\n"
        "*5.* Trancamento de Matrícula\n\n"
        "_Ou digite sua dúvida sobre datas._\n"
        "🔙 *Voltar* para o início."
    ),
    EstadoMenu.SUB_EDITAL: (
        "📋 *Edital PAES 2026*\n\n"
        "*1.* Categorias de vagas (AC, PcD, cotas)\n"
        "*2.* Documentos para inscrição\n"
        "*3.* Cronograma do processo seletivo\n"
        "*4.* Cursos e vagas ofertados\n\n"
        "_Ou digite sua dúvida sobre o edital._\n"
        "🔙 *Voltar* para o início."
    ),
    EstadoMenu.SUB_CONTATOS: (
        "📞 *Contatos e E-mails UEMA*\n\n"
        "*1.* Pró-Reitorias (PROG, PROEXAE, PRPPG...)\n"
        "*2.* Centros Acadêmicos (CECEN, CESB, CESC...)\n"
        "*3.* Coordenações de Curso\n"
        "*4.* TI e CTIC\n\n"
        "_Ou digite o nome do setor que procura._\n"
        "🔙 *Voltar* para o início."
    ),
}

# Opções numéricas de cada submenu expandidas para prompt do LLM
OPCOES_SUBMENU: dict[EstadoMenu, dict[str, str]] = {
    EstadoMenu.SUB_CALENDARIO: {
        "1": "Quais são as datas de matrícula e rematrícula para veteranos e calouros em 2026?",
        "2": "Quando começam e terminam os semestres letivos de 2026?",
        "3": "Quais são os feriados e recessos do calendário acadêmico de 2026?",
        "4": "Quais são as datas de provas, avaliações finais e substitutivas em 2026?",
        "5": "Qual é o prazo para trancamento de matrícula ou de curso em 2026?",
    },
    EstadoMenu.SUB_EDITAL: {
        "1": "Quais são as categorias de vagas do PAES 2026? Explique AC, PcD, BR-PPI, BR-Q, BR-DC e demais cotas.",
        "2": "Quais documentos são necessários para se inscrever no PAES 2026?",
        "3": "Qual é o cronograma do PAES 2026? Datas de inscrição, resultado e matrícula.",
        "4": "Quais cursos e quantas vagas são ofertadas no PAES 2026?",
    },
    EstadoMenu.SUB_CONTATOS: {
        "1": "Quais são os e-mails e telefones das Pró-Reitorias da UEMA?",
        "2": "Quais são os contatos dos centros acadêmicos da UEMA (CECEN, CESB, CESC)?",
        "3": "Quais são os e-mails e telefones das coordenações de curso da UEMA?",
        "4": "Qual é o contato da equipe de TI e do CTIC da UEMA?",
    },
}

# =============================================================================
# Padrões (compilados uma vez)
# =============================================================================

_RE_VOLTAR = re.compile(
    r"^\s*(voltar?|volta|menu|inicio|início|0|cancelar|sair|oi|olá|ola|ajuda|help|start)\s*$",
    re.IGNORECASE,
)

_OPCAO_PARA_ESTADO: dict[str, EstadoMenu] = {
    "1": EstadoMenu.SUB_CALENDARIO,
    "2": EstadoMenu.SUB_EDITAL,
    "3": EstadoMenu.SUB_CONTATOS,
}

_ALIAS_PARA_ESTADO: list[tuple[re.Pattern, EstadoMenu]] = [
    (re.compile(r"calendari|semestre|datas", re.IGNORECASE), EstadoMenu.SUB_CALENDARIO),
    (re.compile(r"edital|paes|vestibular|processo.seletivo", re.IGNORECASE), EstadoMenu.SUB_EDITAL),
    (re.compile(r"contato|email|telefone|e-mail", re.IGNORECASE), EstadoMenu.SUB_CONTATOS),
]


# =============================================================================
# Funções puras
# =============================================================================

def _norm(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode()
    return s.lower().strip()


def processar_mensagem(texto: str, estado: EstadoMenu) -> dict:
    """
    Lógica de menu pura. Retorna um dict com:

      type: "menu_principal" | "submenu" | "llm"
      content: str (texto do menu) | None
      novo_estado: EstadoMenu
      prompt: str (pergunta expandida para o LLM) | None
    """
    txt = texto.strip()

    # ── Voltar / saudação → menu principal ────────────────────────────────────
    if _RE_VOLTAR.match(txt):
        return _ir_para_main()

    # ── No MAIN ───────────────────────────────────────────────────────────────
    if estado == EstadoMenu.MAIN:
        # Opção numérica
        sub = _OPCAO_PARA_ESTADO.get(txt)
        if sub:
            return {"type": "submenu", "content": TEXTO_SUBMENU[sub],
                    "novo_estado": sub, "prompt": None}
        # Alias textual ("edital", "calendário"...)
        for pattern, sub in _ALIAS_PARA_ESTADO:
            if pattern.search(txt):
                return {"type": "submenu", "content": TEXTO_SUBMENU[sub],
                        "novo_estado": sub, "prompt": None}
        # Texto livre → LLM
        return _ir_para_llm(txt, EstadoMenu.MAIN)

    # ── Nos submenus ──────────────────────────────────────────────────────────
    if estado in OPCOES_SUBMENU:
        pergunta = OPCOES_SUBMENU[estado].get(txt)
        if pergunta:
            return _ir_para_llm(pergunta, EstadoMenu.MAIN)
        # Texto livre no submenu → LLM com estado limpo
        return _ir_para_llm(txt, EstadoMenu.MAIN)

    return _ir_para_llm(txt, EstadoMenu.MAIN)


def _ir_para_main() -> dict:
    return {
        "type": "menu_principal",
        "content": MENU_PRINCIPAL,
        "novo_estado": EstadoMenu.MAIN,
        "prompt": None,
    }


def _ir_para_llm(prompt: str, novo_estado: EstadoMenu) -> dict:
    return {
        "type": "llm",
        "content": None,
        "novo_estado": novo_estado,
        "prompt": prompt,
    }