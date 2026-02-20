"""
================================================================================
router_service.py — Roteamento por Intenção (v4 — 3 PDFs)
================================================================================

RESUMO DAS MUDANÇAS NESTA VERSÃO:
  - Rotas ativas: CALENDARIO, EDITAL, CONTATOS
  - SUPORTE comentado (reativar com LLM superior + GLPI funcional)
  - Adicionada rota EDITAL com palavras-chave do processo seletivo PAES
  - Contextos mais específicos por rota para guiar a LLM na escolha da tool
  - montar_prompt_enriquecido() mantido e ajustado para 3 rotas
================================================================================
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


def _normalizar(texto: str) -> str:
    """
    Remove acentos e converte para minúsculas.
    Garante matching robusto independente de como o usuário digitou.
    """
    sem_acento = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    return sem_acento.lower()


class RouterService:
    def __init__(self):
        self.patterns = {

            # ── Opções numéricas do menu principal ───────────────────────────
            # match() garante que o texto INTEIRO seja apenas a opção.
            # Ex: "2 de fevereiro" NÃO deve virar OPCAO_2.
            "OPCAO_1": re.compile(r"^\s*(1|um|calendario|calendario academico)\s*$"),
            "OPCAO_2": re.compile(r"^\s*(2|dois|edital|paes|processo seletivo|vestibular)\s*$"),
            "OPCAO_3": re.compile(r"^\s*(3|tres|contatos?|emails?|telefones?)\s*$"),
            # "OPCAO_4": re.compile(r"^\s*(4|quatro|suporte|ti|glpi|chamado)\s*$"),  ← futuro

            # ── Reset / reinício ─────────────────────────────────────────────
            "RESET": re.compile(
                r"\b(reiniciar|reset|limpar|recomecar|tchau|sair|cancelar|voltar|inicio)\b"
            ),

            # ── Saudações ────────────────────────────────────────────────────
            # Só ativa se a mensagem for APENAS uma saudação (sem pergunta junto).
            # Ex: "oi" → MENU | "oi quando é a prova" → cai no CALENDARIO
            "MENU": re.compile(
                r"^\s*(oi|ola|bom dia|boa tarde|boa noite|ajuda|menu|start|help|"
                r"oi tudo bem|oi boa tarde|oi bom dia|ola tudo bem)\s*$"
            ),

            # ── Intenção: Calendário Acadêmico ───────────────────────────────
            # Palavras de datas e eventos do calendário letivo.
            # Nota: "data" sozinha não está aqui para evitar falso positivo.
            "CALENDARIO": re.compile(
                r"\b(prazo|feriado|prova|matricula|rematricula|semestre|periodo|"
                r"trancamento|calendario|inicio das aulas|termino das aulas|"
                r"retardatario|veterano|calouro|reingresso|avaliacao|substitutiva|"
                r"recesso|defesa|banca|2026\.1|2026\.2|primeiro semestre|"
                r"segundo semestre|aula|letivo)\b"
            ),

            # ── Intenção: Edital PAES 2026 ────────────────────────────────────
            # Palavras ligadas ao processo seletivo, vagas e cotas.
            "EDITAL": re.compile(
                r"\b(edital|paes|vestibular|processo seletivo|inscricao|inscricoes|"
                r"vaga|vagas|cota|cotas|ac|pcd|br-ppi|br-q|br-dc|ir-ppi|cfo-pp|"
                r"ampla concorrencia|rede publica|quilombola|indigena|deficiencia|"
                r"documentos|cronograma|resultado|classificacao|convocacao|"
                r"heteroidentificacao|reserva de vaga|curso ofertado)\b"
            ),

            # ── Intenção: Contatos ────────────────────────────────────────────
            "CONTATOS": re.compile(
                r"\b(contato|email|e-mail|telefone|fone|ramal|prog|proexae|prppg|prad|"
                r"reitoria|ctic|departamento|coordenacao|secretaria|ouvidoria|"
                r"pro-reitoria|pró-reitoria|cecen|cesb|cesc|ccsa|diretor|coordenador|"
                r"central de atendimento|ti da uema|suporte uema)\b"
            ),

            # ── Intenção: Suporte Técnico (COMENTADO — futuro) ────────────────
            # "SUPORTE": re.compile(
            #     r"\b(glpi|chamado|suporte|computador|pc|notebook|impressora|"
            #     r"internet|net|wifi|wi.fi|login|senha|siguema|sistema|acesso|"
            #     r"laboratorio|monitor|teclado|mouse|projetor)\b"
            # ),
        }

        # ── Contextos pré-definidos por rota ─────────────────────────────────
        # Esses textos vão junto com o prompt para guiar a LLM a usar a tool certa.
        self._contextos = {
            "CALENDARIO": (
                "O usuário tem uma dúvida sobre datas ou eventos do calendário acadêmico da UEMA 2026. "
                "Use EXCLUSIVAMENTE a ferramenta 'consultar_calendario_academico'. "
                "Passe palavras-chave específicas como query (ex: 'matricula veteranos 2026.1'). "
                "Nunca invente datas — use apenas o que a ferramenta retornar."
            ),
            "EDITAL": (
                "O usuário tem uma dúvida sobre o Edital do PAES 2026 (processo seletivo da UEMA). "
                "Use EXCLUSIVAMENTE a ferramenta 'consultar_edital_paes_2026'. "
                "Passe termos específicos como query (ex: 'vagas ampla concorrencia', 'documentos inscricao'). "
                "Nunca invente regras ou números de vagas."
            ),
            "CONTATOS": (
                "O usuário quer encontrar um contato, e-mail ou telefone da UEMA. "
                "Use EXCLUSIVAMENTE a ferramenta 'consultar_contatos_uema'. "
                "Passe o nome do setor ou cargo como query (ex: 'PROG pro-reitoria', 'CTIC TI'). "
                "Nunca invente e-mails ou telefones."
            ),
            # "SUPORTE": (        ← futuro
            #     "O usuário precisa de suporte técnico. Colete: tipo do problema, "
            #     "local (sala/bloco) e nome completo. Use 'abrir_chamado_glpi'."
            # ),
            "MENU": (
                "Exibir o menu principal com as opções disponíveis. Não use nenhuma ferramenta."
            ),
            "RESET": (
                "Reiniciar a conversa e exibir o menu principal."
            ),
            "GERAL": (
                "Assunto não identificado claramente. Responda com o que souber "
                "ou oriente o usuário a usar o menu principal para escolher uma área."
            ),
        }

    # =========================================================================
    # Análise principal
    # =========================================================================

    def analisar(self, texto: str, estado_menu: str = "MAIN") -> dict:
        """
        Identifica a intenção do usuário e retorna rota + contexto.

        Parâmetros:
          texto       : mensagem original do usuário
          estado_menu : estado atual do MenuService (evita conflito de rota)

        Retorno:
          {"rota": str, "contexto": str}

        Prioridade:
          1. Opções numéricas (match exato no texto inteiro)
          2. Reset
          3. Saudação (só se estiver no MAIN)
          4. Palavras-chave por área (EDITAL antes de CALENDARIO para evitar
             ambiguidade com "data de inscrição")
          5. Fallback GERAL
        """
        texto_norm = _normalizar(texto.strip())
        logger.debug("🔍 Router | texto: '%s' | estado: %s", texto_norm[:60], estado_menu)

        # ── 1. Opções numéricas ───────────────────────────────────────────────
        for padrao, rota in [
            ("OPCAO_1", "CALENDARIO"),
            ("OPCAO_2", "EDITAL"),
            ("OPCAO_3", "CONTATOS"),
            # ("OPCAO_4", "SUPORTE"),   ← futuro
        ]:
            if self.patterns[padrao].match(texto_norm):
                logger.info("🔢 Rota por opção numérica: %s", rota)
                return {"rota": rota, "contexto": self._contextos[rota]}

        # ── 2. Reset ──────────────────────────────────────────────────────────
        if self.patterns["RESET"].search(texto_norm):
            logger.info("🔄 Rota RESET.")
            return {"rota": "RESET", "contexto": self._contextos["RESET"]}

        # ── 3. Saudação (só no MAIN) ──────────────────────────────────────────
        if self.patterns["MENU"].match(texto_norm) and estado_menu == "MAIN":
            logger.info("👋 Rota MENU (saudação).")
            return {"rota": "MENU", "contexto": self._contextos["MENU"]}

        # ── 4. Palavras-chave por área ────────────────────────────────────────
        # EDITAL antes de CALENDARIO: "data de inscrição do PAES" → EDITAL
        if self.patterns["EDITAL"].search(texto_norm):
            logger.info("📋 Rota EDITAL por palavra-chave.")
            return {"rota": "EDITAL", "contexto": self._contextos["EDITAL"]}

        if self.patterns["CALENDARIO"].search(texto_norm):
            logger.info("📅 Rota CALENDARIO por palavra-chave.")
            return {"rota": "CALENDARIO", "contexto": self._contextos["CALENDARIO"]}

        if self.patterns["CONTATOS"].search(texto_norm):
            logger.info("📞 Rota CONTATOS por palavra-chave.")
            return {"rota": "CONTATOS", "contexto": self._contextos["CONTATOS"]}

        # ── 5. Fallback ───────────────────────────────────────────────────────
        logger.info("🌐 Rota GERAL (fallback).")
        return {"rota": "GERAL", "contexto": self._contextos["GERAL"]}

    # =========================================================================
    # Montagem do prompt enriquecido
    # =========================================================================

    def montar_prompt_enriquecido(
        self,
        texto_usuario: str,
        rota: dict,
        contexto_usuario: dict = None,
    ) -> str:
        """
        Monta o prompt completo para enviar ao agente LLM.

        Combina:
          - Orientação de rota (qual tool usar e como)
          - Dados do usuário se disponíveis (nome, curso)
          - Mensagem original do usuário

        Isso elimina a ambiguidade do modelo: em vez de receber
        só "quando é a prova?", ele recebe contexto completo que
        o direciona para a ferramenta e query corretas.
        """
        linhas = ["[CONTEXTO DO ATENDIMENTO]"]
        linhas.append(f"Área: {rota['rota']}")
        linhas.append(f"Instrução: {rota['contexto']}")

        if contexto_usuario:
            if nome := contexto_usuario.get("nome"):
                linhas.append(f"Nome do usuário: {nome}")
            if curso := contexto_usuario.get("curso"):
                linhas.append(f"Curso: {curso}")

        linhas.append("")
        linhas.append("[MENSAGEM DO USUÁRIO]")
        linhas.append(texto_usuario)

        prompt = "\n".join(linhas)
        logger.debug("📝 Prompt enriquecido:\n%s", prompt)
        return prompt