import redis
from src.config import settings

class MenuService:
    def __init__(self):
        self.r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        
        self.menus = {
            "MAIN": {
                "msg": (
                    "👋 *Olá! Sou o Assistente Virtual da UEMA.*\n"
                    "Por favor, escolha uma opção:\n\n"
                    "📅 *1. Calendário Académico*\n"
                    "🛠️ *2. Suporte Técnico (TI)*\n"
                    "🍔 *3. RU e Transporte*\n"
                    "📞 *4. Contatos e Emails*"
                ),
                "opcoes": {
                    "1": "SUB_CALENDARIO",
                    "2": "SUB_SUPORTE",
                    "3": "SUB_RU",
                    "4": "SUB_CONTATOS"
                }
            },
            "SUB_CALENDARIO": {
                "msg": (
                    "📅 *Calendário Académico*\n"
                    "O que deseja saber?\n\n"
                    "1️⃣ Matrícula/Rematrícula\n"
                    "2️⃣ Feriados e Recessos\n"
                    "3️⃣ Provas e Avaliações\n"
                    "4️⃣ Digitar um mês específico\n\n"
                    "🔙 Digite *Voltar* para o início."
                )
            },
            "SUB_SUPORTE": {
                "msg": (
                    "🛠️ *Suporte Técnico (GLPI)*\n"
                    "Selecione o problema:\n\n"
                    "1️⃣ Internet / Wi-Fi\n"
                    "2️⃣ Hardware / PC\n"
                    "3️⃣ Login / SigUema\n\n"
                    "🔙 Digite *Voltar* para o início."
                )
            }
        }

    def get_user_state(self, user_id: str) -> str:
        return self.r.get(f"menu_state:{user_id}") or "MAIN"

    def set_user_state(self, user_id: str, state: str):
        self.r.setex(f"menu_state:{user_id}", 300, state)

    def clear_user_state(self, user_id: str):
        self.r.delete(f"menu_state:{user_id}")

    def processar_escolha(self, user_id: str, texto: str):
        estado_atual = self.get_user_state(user_id)
        texto_limpo = texto.strip().lower()

        if texto_limpo in ["voltar", "inicio", "menu", "sair"]:
            self.clear_user_state(user_id)
            return {"type": "msg", "content": self.menus["MAIN"]["msg"]}

        if estado_atual == "MAIN":
            proximo = self.menus["MAIN"]["opcoes"].get(texto_limpo)
            if proximo:
                self.set_user_state(user_id, proximo)
                return {"type": "msg", "content": self.menus[proximo]["msg"]}
            return {"type": "action", "prompt": texto} # Se não for opção, deixa a IA decidir

        # Lógica de Submenu -> Transforma número em Prompt para a IA
        if estado_atual == "SUB_CALENDARIO":
            self.clear_user_state(user_id)
            contexto = "Contexto: Calendário Académico UEMA São Luís."
            if texto_limpo == "1": return {"type": "action", "prompt": f"{contexto} Foco: Matrícula e Rematrícula."}
            if texto_limpo == "2": return {"type": "action", "prompt": f"{contexto} Foco: Feriados e Recessos."}
            return {"type": "action", "prompt": texto}

        self.clear_user_state(user_id)
        return {"type": "action", "prompt": texto}