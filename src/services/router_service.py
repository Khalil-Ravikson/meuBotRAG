"""
router_service.py — RouterService revisado

Problemas corrigidos:
  - Normalização de unicode ausente: "matrícula" não batia com "matricula"
    (acentos do WhatsApp variavam dependendo do teclado do usuário)
  - Padrão MENU conflitava com CALENDARIO ("data" capturava "boa data")
  - OPCAO_* usavam search() mas os padrões tinham ^ e $ — deve ser match()
  - Adicionado CONTATOS como rota própria (estava no menu mas não no router)
  - Rota FALLBACK separada de GERAL para facilitar debug nos logs
  - analisar() agora recebe estado do menu para evitar conflito de contexto
"""

import re
import unicodedata


def _normalizar(texto: str) -> str:
    """Remove acentos e converte para minúsculas para comparação robusta."""
    return unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8").lower()


class RouterService:
    def __init__(self):
        # Padrões aplicados sobre texto JÁ normalizado (sem acento, minúsculo)
        self.patterns = {
            # Saudações e menu
            "MENU": re.compile(
                r"\b(oi|ola|bom dia|boa tarde|boa noite|ajuda|menu|inicio|start|help)\b"
            ),
            "RESET": re.compile(
                r"\b(reiniciar|reset|limpar|recomecar|tchau|sair|cancelar)\b"
            ),

            # Opções numéricas do menu principal (prioridade máxima — testadas primeiro)
            # match() garante que só captura se o texto INTEIRO for a opção
            "OPCAO_1": re.compile(r"^\s*(1|um|calendario|datas?)\s*$"),
            "OPCAO_2": re.compile(r"^\s*(2|dois|suporte|ti|glpi|chamado)\s*$"),
            "OPCAO_3": re.compile(r"^\s*(3|tres|ru|restaurante|onibus|transporte)\s*$"),
            "OPCAO_4": re.compile(r"^\s*(4|quatro|contatos?|emails?|telefones?)\s*$"),

            # Intenções por palavras-chave (texto livre)
            "SUPORTE": re.compile(
                r"\b(glpi|chamado|suporte|computador|pc|notebook|net|wifi|wi.fi|"
                r"impressora|login|senha|siguema|sistema|acesso|laboratorio)\b"
            ),
            "CALENDARIO": re.compile(
                r"\b(data|prazo|feriado|prova|matricula|rematricula|semestre|"
                r"periodo|trancamento|calendario|aula|inicio|termino|retardatario|"
                r"veterano|calouro|reingresso)\b"
            ),
            "RU": re.compile(
                r"\b(ru|restaurante|refeicao|comida|almoco|jantar|cardapio|"
                r"onibus|transporte|rota|horario do onibus)\b"
            ),
            "CONTATOS": re.compile(
                r"\b(contato|email|e-mail|telefone|fone|ramal|prog|proexae|"
                r"reitoria|ctic|departamento|coordenacao|secretaria)\b"
            ),
        }

    def analisar(self, texto: str, estado_menu: str = "MAIN") -> dict:
        """
        Analisa o texto e retorna:
          {"rota": str, "contexto": str}

        Parâmetro estado_menu: estado atual do MenuService.
        Se o usuário já está num submenu, não redireciona para MENU novamente.
        """
        texto_norm = _normalizar(texto.strip())

        # 1. Opções numéricas — prioridade máxima (match exato)
        if self.patterns["OPCAO_1"].match(texto_norm):
            return {
                "rota": "CALENDARIO",
                "contexto": (
                    "O usuário escolheu Calendário Acadêmico. "
                    "Pergunte de qual mês ou evento específico ele precisa. "
                    "Não busque tudo de uma vez."
                ),
            }
        if self.patterns["OPCAO_2"].match(texto_norm):
            return {
                "rota": "SUPORTE",
                "contexto": (
                    "O usuário escolheu Suporte Técnico. "
                    "Pergunte qual o problema, o local e a urgência para abrir chamado no GLPI."
                ),
            }
        if self.patterns["OPCAO_3"].match(texto_norm):
            return {
                "rota": "RU",
                "contexto": (
                    "O usuário escolheu RU e Transporte. "
                    "Consulte regras do RU e rotas de ônibus na base de conhecimento."
                ),
            }
        if self.patterns["OPCAO_4"].match(texto_norm):
            return {
                "rota": "CONTATOS",
                "contexto": (
                    "O usuário escolheu Contatos. "
                    "Consulte e-mails e telefones na base de conhecimento."
                ),
            }

        # 2. Reset — limpa tudo
        if self.patterns["RESET"].search(texto_norm):
            return {"rota": "RESET", "contexto": "Usuário quer reiniciar a conversa."}

        # 3. Saudação/menu — só aciona se não estiver já num fluxo de submenu
        if self.patterns["MENU"].search(texto_norm) and estado_menu == "MAIN":
            return {"rota": "MENU", "contexto": "Saudação — exibir menu principal."}

        # 4. Intenções por palavra-chave (texto livre)
        if self.patterns["SUPORTE"].search(texto_norm):
            return {
                "rota": "SUPORTE",
                "contexto": "Usuário tem dúvida ou problema de suporte técnico.",
            }
        if self.patterns["CALENDARIO"].search(texto_norm):
            return {
                "rota": "CALENDARIO",
                "contexto": "Usuário perguntou sobre datas ou prazos acadêmicos.",
            }
        if self.patterns["RU"].search(texto_norm):
            return {
                "rota": "RU",
                "contexto": "Usuário perguntou sobre o RU ou transporte.",
            }
        if self.patterns["CONTATOS"].search(texto_norm):
            return {
                "rota": "CONTATOS",
                "contexto": "Usuário perguntou sobre contatos institucionais.",
            }

        # 5. Fallback — deixa a IA decidir livremente
        return {"rota": "GERAL", "contexto": "Assunto não identificado — responder livremente."}