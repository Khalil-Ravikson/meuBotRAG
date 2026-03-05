import re
from dataclasses import dataclass
from typing import Optional

# ────────────────────────────────────────────────────────────────────────
# ESTRUTURA DE DADOS
# ────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    bloquear: bool          # Se True, interrompe o fluxo e retorna a 'resposta' imediatamente
    resposta: Optional[str] # A resposta enlatada (greeter, erro, aviso) ou None
    precisa_rag: bool       # Sinalizador para o Self-RAG (True = buscar no Redis, False = LLM direto)

# ────────────────────────────────────────────────────────────────────────
# SERVIÇO DE GUARDRAILS (AJUSTÁVEL E ESCALÁVEL)
# ────────────────────────────────────────────────────────────────────────

class GuardrailService:
    def __init__(self):
        # BLOCO 1: Padrões de Saudação e Ajuda (Case Insensitive)
        self.regex_saudacao = re.compile(
            r'^(oi|olá|ola|bom dia|boa tarde|boa noite|opa|e aí|eae|tudo bem|ola tudo bem)\b', 
            re.IGNORECASE
        )
        self.regex_ajuda = re.compile(
            r'^(ajuda|menu|o que voce faz|o que você faz|como funciona|socorro|\?)$', 
            re.IGNORECASE
        )

        # BLOCO 2: Padrões Ofensivos ou Fora de Escopo (Blocklist 0 tokens)
        # Nota: Expanda esta lista conforme o uso dos alunos. 
        self.regex_ofensivo = re.compile(
            r'\b(idiota|burro|inútil|merda|porra|caralho|vsf|fdp)\b', 
            re.IGNORECASE
        )
        
        # Padrões que indicam claramente que o usuário quer falar de algo fora do escopo da universidade
        self.regex_fora_escopo = re.compile(
            r'\b(futebol|brasileirão|aposta|tigrinho|bet365|receita de|como cozinhar|politica|bolsonaro|lula)\b', 
            re.IGNORECASE
        )

        # BLOCO 3: Heurística Self-RAG (Termos que exigem busca nos documentos)
        # Se a pergunta tiver esses termos, é 100% de certeza que precisa de RAG.
        self.regex_termos_rag = re.compile(
            r'\b(edital|calendário|matrícula|reitor|curso|disciplina|prazo|data|documento|ru|restaurante universitário|bolsa|auxílio|sigaa|nota|histórico|diploma|tcc|estágio|biblioteca)\b',
            re.IGNORECASE
        )

        # Respostas Padrão (Podem virar variáveis de ambiente no futuro)
        self.msg_boas_vindas = (
            "👋 Olá! Eu sou o assistente virtual da instituição.\n\n"
            "Posso te ajudar com:\n"
            "📅 Calendário Acadêmico\n"
            "📄 Editais e Prazos\n"
            "🍽️ Regras do RU\n"
            "🎓 Dúvidas sobre cursos\n\n"
            "Como posso te ajudar hoje?"
        )
        
        self.msg_bloqueio_ofensivo = (
            "⚠️ Por favor, vamos manter o respeito. Sou um assistente virtual focado em ajudar com assuntos acadêmicos e institucionais. Como posso ser útil?"
        )
        
        self.msg_bloqueio_escopo = (
            "🤖 Desculpe, mas eu fui treinado exclusivamente para responder a perguntas sobre a nossa instituição (editais, calendário, cursos, etc.). Não consigo te ajudar com esse outro assunto."
        )

    def analisar(self, mensagem: str) -> GuardrailResult:
        """
        Avalia a mensagem do usuário e retorna a ação de Guardrail apropriada.
        """
        msg_limpa = mensagem.strip()

        # 1. Verificar Blocklist (Ofensivo / Fora de Escopo)
        if self.regex_ofensivo.search(msg_limpa):
            return GuardrailResult(bloquear=True, resposta=self.msg_bloqueio_ofensivo, precisa_rag=False)
            
        if self.regex_fora_escopo.search(msg_limpa):
            return GuardrailResult(bloquear=True, resposta=self.msg_bloqueio_escopo, precisa_rag=False)

        # 2. Verificar Greeter (Saudações isoladas e Pedidos de Ajuda)
        # Se a mensagem for APENAS "oi" ou "ajuda", respondemos com o menu.
        if self.regex_ajuda.match(msg_limpa) or (self.regex_saudacao.match(msg_limpa) and len(msg_limpa.split()) <= 3):
            return GuardrailResult(bloquear=True, resposta=self.msg_boas_vindas, precisa_rag=False)

        # 3. Detector Self-RAG
        # Se passou pelos bloqueios e não é apenas um "oi", avaliamos se precisa do banco vetorial.
        precisa_rag = False
        
        # Heurística 1: Tem palavras-chave de documentos institucionais?
        if self.regex_termos_rag.search(msg_limpa):
            precisa_rag = True
        # Heurística 2: Perguntas estruturadas geralmente precisam de RAG
        elif any(palavra in msg_limpa.lower() for palavra in ["qual", "quando", "onde", "como faço", "é verdade", "documento"]):
            precisa_rag = True
            
        # Retorna o fluxo normal (bloquear=False), deixando o LLM/Agente assumir
        return GuardrailResult(bloquear=False, resposta=None, precisa_rag=precisa_rag)

# Instância global para ser importada e usada no projeto
guardrails = GuardrailService()