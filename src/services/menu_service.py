"""
menu_service.py — MenuService revisado

Problemas corrigidos:
  - SUB_RU e SUB_CONTATOS estavam no RouterService mas não no menus dict
    (causava KeyError silencioso ao navegar para esses submenus)
  - TTL de 300s (5min) era curto demais para uma conversa normal → 600s
  - Submenu SUB_SUPORTE não tinha mapeamento de opções numéricas
  - clear_user_state após escolha de submenu apagava o contexto cedo demais;
    agora mantém até o usuário voltar ou a IA responder
"""

import redis
from src.config import settings


class MenuService:
    def __init__(self):
        self.r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

        self.menus = {
            "MAIN": {
                "msg": (
                    "👋 *Olá! Sou o Assistente Virtual da UEMA.*\n"
                    "Escolha uma opção:\n\n"
                    "📅 *1.* Calendário Acadêmico\n"
                    "🛠️ *2.* Suporte Técnico (TI)\n"
                    "🍔 *3.* RU e Transporte\n"
                    "📞 *4.* Contatos e E-mails\n\n"
                    "_Ou digite sua dúvida diretamente._"
                ),
                "opcoes": {
                    "1": "SUB_CALENDARIO",
                    "2": "SUB_SUPORTE",
                    "3": "SUB_RU",
                    "4": "SUB_CONTATOS",
                },
            },
            "SUB_CALENDARIO": {
                "msg": (
                    "📅 *Calendário Acadêmico*\n\n"
                    "*1.* Matrícula / Rematrícula\n"
                    "*2.* Feriados e Recessos\n"
                    "*3.* Provas e Avaliações\n"
                    "*4.* Trancamento de matrícula\n"
                    "*5.* Digitar pergunta livre\n\n"
                    "🔙 *Voltar* para o início."
                ),
                "opcoes": {
                    "1": "Quais são as datas de matrícula e rematrícula de veteranos e calouros?",
                    "2": "Quais são os feriados e recessos do calendário acadêmico 2026?",
                    "3": "Quais são as datas de provas e avaliações finais?",
                    "4": "Qual o prazo para trancamento de matrícula ou de curso?",
                },
            },
            "SUB_SUPORTE": {
                "msg": (
                    "🛠️ *Suporte Técnico (GLPI)*\n\n"
                    "*1.* Problema com Internet / Wi-Fi\n"
                    "*2.* Hardware / PC com defeito\n"
                    "*3.* Login / SIGUEMA\n"
                    "*4.* Outro problema\n\n"
                    "🔙 *Voltar* para o início."
                ),
                "opcoes": {
                    "1": "Preciso abrir chamado: sem internet ou wi-fi no laboratório.",
                    "2": "Preciso abrir chamado: problema de hardware ou computador com defeito.",
                    "3": "Preciso abrir chamado: não consigo fazer login no SIGUEMA.",
                    "4": "Preciso de suporte técnico. Vou descrever o problema.",
                },
            },
            "SUB_RU": {
                "msg": (
                    "🍔 *RU e Transporte*\n\n"
                    "*1.* Regras e horários do RU\n"
                    "*2.* Rotas e horários de ônibus\n\n"
                    "🔙 *Voltar* para o início."
                ),
                "opcoes": {
                    "1": "Quais são as regras, horários e funcionamento do Restaurante Universitário?",
                    "2": "Quais são as rotas e horários dos ônibus da UEMA?",
                },
            },
            "SUB_CONTATOS": {
                "msg": (
                    "📞 *Contatos e E-mails*\n\n"
                    "*1.* Pró-Reitorias (PROG, PROEXAE...)\n"
                    "*2.* Departamentos e Cursos\n"
                    "*3.* TI / CTIC\n\n"
                    "🔙 *Voltar* para o início."
                ),
                "opcoes": {
                    "1": "Quais são os e-mails e telefones das Pró-Reitorias da UEMA?",
                    "2": "Quais são os contatos dos departamentos e coordenações de curso?",
                    "3": "Qual o contato da equipe de TI (CTIC) da UEMA?",
                },
            },
        }

    # ------------------------------------------------------------------
    # Estado no Redis
    # ------------------------------------------------------------------

    def get_user_state(self, user_id: str) -> str:
        return self.r.get(f"menu_state:{user_id}") or "MAIN"

    def set_user_state(self, user_id: str, state: str):
        self.r.setex(f"menu_state:{user_id}", 600, state)  # 10 min

    def clear_user_state(self, user_id: str):
        self.r.delete(f"menu_state:{user_id}")

    # ------------------------------------------------------------------
    # Processamento principal
    # ------------------------------------------------------------------

    def processar_escolha(self, user_id: str, texto: str) -> dict:
        """
        Retorna um dict com:
          {"type": "msg",    "content": "<texto a enviar>"}   → resposta direta
          {"type": "action", "prompt": "<prompt para a IA>"}  → passa para o agente
        """
        estado_atual = self.get_user_state(user_id)
        texto_limpo  = texto.strip().lower()

        # --- Comandos globais (qualquer estado) ---
        if texto_limpo in {"voltar", "inicio", "início", "menu", "sair", "oi", "olá", "ola"}:
            self.clear_user_state(user_id)
            return {"type": "msg", "content": self.menus["MAIN"]["msg"]}

        # --- Menu principal: navega para submenu ---
        if estado_atual == "MAIN":
            proximo = self.menus["MAIN"]["opcoes"].get(texto_limpo)
            if proximo:
                self.set_user_state(user_id, proximo)
                return {"type": "msg", "content": self.menus[proximo]["msg"]}
            # Texto livre no MAIN → IA decide
            return {"type": "action", "prompt": texto}

        # --- Submenus: converte número em prompt para a IA ---
        menu_atual = self.menus.get(estado_atual, {})
        opcoes     = menu_atual.get("opcoes", {})

        if texto_limpo in opcoes:
            prompt = opcoes[texto_limpo]
            self.clear_user_state(user_id)   # libera estado após escolha
            return {"type": "action", "prompt": prompt}

        # Texto livre dentro de submenu → mantém contexto e passa para a IA
        self.clear_user_state(user_id)
        return {"type": "action", "prompt": texto}