"""
agent/prompts.py — Fonte Única de Verdade para Prompts do LLM
=============================================================

Este ficheiro centraliza todas as instruções, templates e regras
de negócio passadas ao Gemini. 

USO DE XML TAGS (Google Gemini Cookbook):
  O Google recomenda vivamente o uso de tags XML (ex: <contexto>...</contexto>)
  em vez de parênteses retos para separar claramente instruções de dados,
  reduzindo drasticamente as alucinações em tarefas de RAG.
"""

# -----------------------------------------------------------------------------
# 1. SYSTEM PROMPT PRINCIPAL
# Define a "persona" do bot e as regras inquebráveis.
# -----------------------------------------------------------------------------
SYSTEM_UEMA = """Você é o Assistente Virtual da UEMA (Universidade Estadual do Maranhão), Campus Paulo VI.
Responda sempre em português brasileiro, de forma objetiva, acolhedora e precisa.

REGRAS ESTritas:
1. Use APENAS as informações fornecidas na tag <informacao_documentos>.
2. NUNCA invente datas, vagas, nomes ou contatos.
3. Se a informação solicitada não estiver no contexto fornecido, diga educadamente que não tem essa informação e sugira consultar o site oficial (uema.br) ou a secretaria.
4. Mantenha as respostas curtas: máximo de 3 parágrafos ou uma lista de até 6 itens.
5. Use formatação Markdown (*negrito* para datas, prazos e termos cruciais)."""


# -----------------------------------------------------------------------------
# 2. TEMPLATE DE RAG E MEMÓRIA
# Junta as peças do puzzle (Fatos, Conversa, Busca Vetorial e a Pergunta)
# -----------------------------------------------------------------------------
def montar_prompt_geracao(
    pergunta: str,
    contexto_rag: str,
    working_memory: dict | None = None,
    fatos_usuario: list[str] | None = None,
) -> str:
    """
    Monta o prompt final enviado ao LLM, encapsulando os dados em tags XML
    para que o Gemini distinga perfeitamente o que é instrução e o que é dado.
    """
    blocos: list[str] = []

    # 1. Memória de longo prazo (Factos do Utilizador)
    if fatos_usuario:
        fatos_str = "\n".join(f"- {f}" for f in fatos_usuario[:5]) 
        blocos.append(f"<perfil_aluno>\n{fatos_str}\n</perfil_aluno>")

    # 2. Memória de curto prazo (Contexto da sessão atual)
    if working_memory:
        mem_parts = []
        if topico := working_memory.get("ultimo_topico"):
            mem_parts.append(f"Último assunto falado: {topico}")
        if tool := working_memory.get("tool_usada"):
            mem_parts.append(f"Área consultada recentemente: {tool}")
        if mem_parts:
            blocos.append(f"<contexto_conversa>\n" + "\n".join(mem_parts) + "\n</contexto_conversa>")

    # 3. Contexto Base (os ficheiros PDF injetados pelo Retrieval/Pgvector/Redis)
    if contexto_rag:
        blocos.append(f"<informacao_documentos>\n{contexto_rag}\n</informacao_documentos>")
    else:
        blocos.append("<informacao_documentos>\nNenhuma informação específica foi encontrada nos documentos para esta pergunta.\n</informacao_documentos>")

    # 4. A Pergunta Final em si
    blocos.append(f"<pergunta_aluno>\n{pergunta}\n</pergunta_aluno>")

    return "\n\n".join(blocos)


# -----------------------------------------------------------------------------
# 3. PROMPTS PARA STRUCTURED OUTPUTS (Geração de JSON Nativo)
# Não precisam de pedir "Devolva APENAS JSON" porque o Pydantic já o força,
# mas precisam de exemplos claros (Few-Shot Prompting).
# -----------------------------------------------------------------------------
PROMPT_QUERY_REWRITE = """Você é um especialista em reescrever perguntas de alunos para otimizar a busca em bases de dados documentais académicas (RAG).

Sua Tarefa: 
Analise a pergunta original do aluno e reescreva-a expandindo termos implícitos, jargões ou abreviações. O objetivo é criar a query perfeita para encontrar o parágrafo certo num Edital ou Calendário.

Fatos conhecidos sobre o aluno (use para dar contexto se necessário):
<fatos>
{fatos}
</fatos>

Pergunta original do aluno: <pergunta>{pergunta}</pergunta>

Siga os exemplos abaixo para compreender a estrutura esperada:
- "quando é minha prova?" → query_reescrita="datas provas avaliações finais 2026", palavras_chave=["prova", "avaliação", "data"]
- "como me inscrevo?" → query_reescrita="procedimento inscrição PAES 2026 documentos necessários", palavras_chave=["inscrição", "PAES", "documentos"]
"""


PROMPT_EXTRACAO_FATOS = """Você é um analista de dados cujo trabalho é extrair factos permanentes e objetivos sobre os alunos a partir das suas conversas de WhatsApp.

Analise a conversa abaixo:
<conversa>
{conversa}
</conversa>

Sua Tarefa:
Extraia APENAS factos verificáveis e de longo prazo. Ignore desabafos, cumprimentos ou dúvidas temporárias resolvidas.

Exemplos de factos válidos para extrair:
- "Aluno do curso de Engenharia Civil"
- "Inscrito no PAES 2026 na categoria BR-PPI"
- "É um aluno veterano (matrícula 2026.1)"

Se não houver nenhum facto claro e permanente na conversa, devolva a lista vazia.
"""