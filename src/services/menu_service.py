"""
================================================================================
menu_service.py — Serviço de Menu por Estado (v4 — 3 PDFs)
================================================================================

RESUMO DAS MUDANÇAS NESTA VERSÃO:
  - Foco em 3 fontes de informação: Calendário, Edital PAES 2026 e Contatos
  - Suporte técnico / GLPI comentado (será reativado com LLM superior)
  - Email e fila comentados (idem)
  - Menus enxutos e diretos, sem opções que o sistema ainda não suporta bem
  - Contexto do usuário mantido via Redis para enriquecer prompts da LLM
================================================================================
"""

import json
import logging
import redis
from src.config import settings

logger = logging.getLogger(__name__)


class MenuService:
    def __init__(self):
        self.r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

        # TTL das chaves Redis
        self.TTL_ESTADO   = 1800   # 30 min de inatividade reseta o menu
        self.TTL_CONTEXTO = 3600   # contexto do usuário dura 1h

        # ── Definição dos menus ──────────────────────────────────────────────
        # Cada menu tem:
        #   "msg"    → texto enviado diretamente ao usuário (sem LLM)
        #   "opcoes" → mapeamento número → próximo estado OU prompt para a LLM
        self.menus = {

            # ── Menu principal ───────────────────────────────────────────────
            "MAIN": {
                "msg": (
                    "👋 *Olá! Sou o Assistente Virtual da UEMA.*\n\n"
                    "Escolha uma opção:\n\n"
                    "📅 *1.* Calendário Acadêmico\n"
                    "📋 *2.* Edital PAES 2026\n"
                    "📞 *3.* Contatos e E-mails\n\n"
                    # "🛠️ *4.* Suporte Técnico (TI)  ← em breve\n"
                    "_Ou digite sua dúvida diretamente._"
                ),
                "opcoes": {
                    "1": "SUB_CALENDARIO",
                    "2": "SUB_EDITAL",
                    "3": "SUB_CONTATOS",
                    # "4": "SUB_SUPORTE",   ← comentado até ter LLM superior
                },
            },

            # ── Submenu: Calendário Acadêmico ────────────────────────────────
            "SUB_CALENDARIO": {
                "msg": (
                    "📅 *Calendário Acadêmico 2026*\n\n"
                    "*1.* Matrícula e Rematrícula\n"
                    "*2.* Início e Fim de Semestre\n"
                    "*3.* Feriados e Recessos\n"
                    "*4.* Provas e Avaliações\n"
                    "*5.* Trancamento de Matrícula\n\n"
                    "_Ou digite sua dúvida sobre datas._\n"
                    "🔙 *Voltar* para o início."
                ),
                "opcoes": {
                    "1": "Quais são as datas de matrícula e rematrícula para veteranos e calouros em 2026?",
                    "2": "Quando começam e terminam os semestres letivos de 2026?",
                    "3": "Quais são os feriados e recessos do calendário acadêmico de 2026?",
                    "4": "Quais são as datas de provas, avaliações finais e substitutivas em 2026?",
                    "5": "Qual é o prazo para trancamento de matrícula ou de curso em 2026?",
                },
            },

            # ── Submenu: Edital PAES 2026 ────────────────────────────────────
            "SUB_EDITAL": {
                "msg": (
                    "📋 *Edital PAES 2026*\n\n"
                    "*1.* Categorias de vagas (AC, PcD, cotas)\n"
                    "*2.* Documentos para inscrição\n"
                    "*3.* Cronograma do processo seletivo\n"
                    "*4.* Cursos e vagas ofertados\n\n"
                    "_Ou digite sua dúvida sobre o edital._\n"
                    "🔙 *Voltar* para o início."
                ),
                "opcoes": {
                    "1": "Quais são as categorias de vagas do PAES 2026? Explique AC, PcD, BR-PPI, BR-Q, BR-DC e demais cotas.",
                    "2": "Quais documentos são necessários para se inscrever no PAES 2026?",
                    "3": "Qual é o cronograma do PAES 2026? Datas de inscrição, resultado e matrícula.",
                    "4": "Quais cursos e quantas vagas são ofertadas no PAES 2026?",
                },
            },

            # ── Submenu: Contatos ────────────────────────────────────────────
            "SUB_CONTATOS": {
                "msg": (
                    "📞 *Contatos e E-mails UEMA*\n\n"
                    "*1.* Pró-Reitorias (PROG, PROEXAE, PRPPG...)\n"
                    "*2.* Centros Acadêmicos (CECEN, CESB, CESC...)\n"
                    "*3.* Coordenações de Curso\n"
                    "*4.* TI e CTIC\n\n"
                    "_Ou digite o nome do setor que procura._\n"
                    "🔙 *Voltar* para o início."
                ),
                "opcoes": {
                    "1": "Quais são os e-mails e telefones das Pró-Reitorias da UEMA?",
                    "2": "Quais são os contatos dos centros acadêmicos da UEMA (CECEN, CESB, CESC)?",
                    "3": "Quais são os e-mails e telefones das coordenações de curso da UEMA?",
                    "4": "Qual é o contato da equipe de TI e do CTIC da UEMA?",
                },
            },

            # ── Submenu: Suporte Técnico (COMENTADO — futuro) ─────────────────
            # "SUB_SUPORTE": {
            #     "msg": (
            #         "🛠️ *Suporte Técnico (GLPI)*\n\n"
            #         "*1.* Problema com Internet ou Wi-Fi\n"
            #         "*2.* Computador ou Hardware com defeito\n"
            #         "*3.* Problema de Login no SIGUEMA\n"
            #         "*4.* Outro problema de TI\n\n"
            #         "🔙 *Voltar* para o início."
            #     ),
            #     "opcoes": {
            #         "1": "Preciso abrir chamado no GLPI: sem internet ou wi-fi no laboratório.",
            #         "2": "Preciso abrir chamado no GLPI: problema de hardware ou computador com defeito.",
            #         "3": "Preciso abrir chamado no GLPI: não consigo fazer login no SIGUEMA.",
            #         "4": "Preciso de suporte técnico. Vou descrever o meu problema.",
            #     },
            # },
        }

        # Palavras que sempre voltam para o MAIN
        self.PALAVRAS_RESET = {
            "voltar", "inicio", "início", "menu", "sair",
            "oi", "olá", "ola", "ajuda", "help", "start",
        }

    # =========================================================================
    # Estado no Redis
    # =========================================================================

    def get_user_state(self, user_id: str) -> str:
        return self.r.get(f"menu_state:{user_id}") or "MAIN"

    def set_user_state(self, user_id: str, state: str) -> None:
        self.r.setex(f"menu_state:{user_id}", self.TTL_ESTADO, state)
        logger.debug("🗂️  Estado [%s] → %s", user_id, state)

    def clear_user_state(self, user_id: str) -> None:
        self.r.delete(f"menu_state:{user_id}")
        logger.debug("🗑️  Estado [%s] limpo.", user_id)

    # =========================================================================
    # Contexto persistente do usuário
    # =========================================================================

    def get_user_context(self, user_id: str) -> dict:
        """Retorna dados persistentes do usuário (nome, curso, última intenção)."""
        raw = self.r.get(f"user_ctx:{user_id}")
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return {}

    def set_user_context(self, user_id: str, dados: dict) -> None:
        """Atualiza campos do contexto sem sobrescrever os existentes (merge)."""
        ctx = self.get_user_context(user_id)
        ctx.update(dados)
        self.r.setex(f"user_ctx:{user_id}", self.TTL_CONTEXTO, json.dumps(ctx))
        logger.debug("💾 Contexto [%s]: %s", user_id, ctx)

    # =========================================================================
    # Processamento principal
    # =========================================================================

    def processar_escolha(self, user_id: str, texto: str) -> dict:
        """
        Processa a mensagem e retorna a ação a tomar.

        Retorno:
          {"type": "msg",    "content": str}
            → Envia texto diretamente ao usuário, sem passar pela LLM.

          {"type": "action", "prompt": str, "contexto_extra": dict}
            → Passa para RagService com metadados de rota.
        """
        estado_atual = self.get_user_state(user_id)
        texto_limpo  = texto.strip().lower()

        logger.debug("📩 [%s] estado='%s' msg='%s'", user_id, estado_atual, texto_limpo[:60])

        # ── Reset global ──────────────────────────────────────────────────────
        if texto_limpo in self.PALAVRAS_RESET:
            self.clear_user_state(user_id)
            return {"type": "msg", "content": self.menus["MAIN"]["msg"]}

        # ── Menu principal → navega para submenu ──────────────────────────────
        if estado_atual == "MAIN":
            proximo = self.menus["MAIN"]["opcoes"].get(texto_limpo)
            if proximo:
                self.set_user_state(user_id, proximo)
                self.set_user_context(user_id, {"ultima_intencao": proximo})
                return {"type": "msg", "content": self.menus[proximo]["msg"]}
            # Texto livre no MAIN → IA
            return {
                "type": "action",
                "prompt": texto,
                "contexto_extra": {"rota": "GERAL", "estado_menu": "MAIN"},
            }

        # ── Submenus → opção numérica vira prompt para a LLM ─────────────────
        menu_atual = self.menus.get(estado_atual, {})
        opcoes     = menu_atual.get("opcoes", {})

        # Mapa de submenu → rota para o Router
        mapa_rota = {
            "SUB_CALENDARIO": "CALENDARIO",
            "SUB_EDITAL":     "EDITAL",
            "SUB_CONTATOS":   "CONTATOS",
            # "SUB_SUPORTE":  "SUPORTE",   ← comentado
        }

        if texto_limpo in opcoes:
            prompt   = opcoes[texto_limpo]
            rota     = mapa_rota.get(estado_atual, "GERAL")
            self.clear_user_state(user_id)
            self.set_user_context(user_id, {"ultima_intencao": rota})
            logger.info("🤖 [%s] Submenu '%s' → LLM com rota '%s'", user_id, estado_atual, rota)
            return {
                "type": "action",
                "prompt": prompt,
                "contexto_extra": {"rota": rota, "estado_menu": estado_atual},
            }

        # Texto livre em submenu → mantém contexto da área e passa para a LLM
        rota_livre = mapa_rota.get(estado_atual, "GERAL")
        self.clear_user_state(user_id)
        return {
            "type": "action",
            "prompt": texto,
            "contexto_extra": {"rota": rota_livre, "estado_menu": estado_atual},
        }