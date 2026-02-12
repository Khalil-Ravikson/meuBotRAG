import re
import unicodedata

class RouterService:
    def __init__(self):
        self.patterns = {
            # 1. MENU e RESET
            "MENU": re.compile(r"(?i)\b(oi|ol[áa]|bom dia|boa tarde|ajuda|menu|in[ií]cio)\b"),
            "RESET": re.compile(r"(?i)\b(reiniciar|reset|limpar|tchau)\b"),
            
            # 2. NUMERAÇÃO DO MENU (AQUI ESTÁ A CORREÇÃO!)
            # Captura se o usuário digitar apenas "1", "1.", "opcao 1", etc.
            "OPCAO_1": re.compile(r"^\s*(1|um|calend[áa]rio)\.?\s*$", re.IGNORECASE),
            "OPCAO_2": re.compile(r"^\s*(2|dois|suporte|glpi)\.?\s*$", re.IGNORECASE),
            "OPCAO_3": re.compile(r"^\s*(3|tr[êe]s|ru|onibus)\.?\s*$", re.IGNORECASE),
            "OPCAO_4": re.compile(r"^\s*(4|quatro|contatos|email)\.?\s*$", re.IGNORECASE),

            # 3. INTENÇÕES GERAIS (Regex antigo continua aqui...)
            "SUPORTE": re.compile(r"(?i)(glpi|chamado|suporte|computador|pc|net|wifi|impressora|login|senha)"),
            "CALENDARIO": re.compile(r"(?i)(data|prazo|feriado|prova|matricula|semestre)"),
            "RU": re.compile(r"(?i)(ru|restaurante|comida|almoço|jantar|cardapio)"),
        }

    def analisar(self, texto: str) -> dict:
        texto_limpo = texto.strip()

        # --- CHECAGEM DE MENU NUMÉRICO (Prioridade Máxima) ---
        if self.patterns["OPCAO_1"].search(texto_limpo):
            return {
                "rota": "CALENDARIO", 
                "contexto": "O usuário escolheu Opção 1 (Calendário). ELE QUER SABER DATAS. Pergunte de qual mês ou evento ele precisa. NÃO BUSQUE TUDO DE UMA VEZ."
            }
        
        if self.patterns["OPCAO_2"].search(texto_limpo):
            return {
                "rota": "SUPORTE", 
                "contexto": "O usuário escolheu Opção 2 (Suporte Técnico). Pergunte qual é o problema, o local e a urgência para abrir chamado."
            }

        if self.patterns["OPCAO_3"].search(texto_limpo):
            return {
                "rota": "RU", 
                "contexto": "O usuário escolheu Opção 3 (RU/Ônibus). Consulte as regras do RU e rotas de ônibus na base de conhecimento."
            }

        if self.patterns["OPCAO_4"].search(texto_limpo):
            return {
                "rota": "CONTATOS", 
                "contexto": "O usuário escolheu Opção 4 (Contatos). Consulte a lista de telefones e emails."
            }

        # --- CHECAGEM PADRÃO ---
        if self.patterns["RESET"].search(texto_limpo):
            return {"rota": "RESET", "contexto": "Reset"}
            
        if self.patterns["MENU"].search(texto_limpo):
            return {"rota": "MENU", "contexto": "Menu"}

        # Se não for número, tenta identificar por palavra-chave
        if self.patterns["SUPORTE"].search(texto_limpo):
            return {"rota": "SUPORTE", "contexto": "Foco em suporte técnico."}
            
        if self.patterns["CALENDARIO"].search(texto_limpo):
             return {"rota": "CALENDARIO", "contexto": "Foco em datas acadêmicas."}

        return {"rota": "GERAL", "contexto": "Assunto geral."}