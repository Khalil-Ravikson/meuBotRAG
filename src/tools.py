from langchain_core.tools import tool
import random

# --- FERRAMENTA 1: ABRIR CHAMADO ---
@tool
def abrir_chamado_glpi(titulo: str, usuario: str, descricao: str):
    """
    Use esta ferramenta quando o usuário confirmar que quer abrir um chamado técnico ou relatar um problema.
    Argumentos necessários: titulo (resumo), usuario (quem pediu) e descricao (detalhes).
    """
    print(f"🔧 [TOOL] Abrindo chamado GLPI: '{titulo}' para {usuario}")
    
    # AQUI VOCÊ COLOCARIA O CÓDIGO REAL (requests.post para API do GLPI)
    # Vamos simular um ID aleatório:
    ticket_id = random.randint(5000, 9999)
    
    return f"Chamado #{ticket_id} criado com sucesso! A equipe de TI vai verificar: {descricao}"

# --- FERRAMENTA 2: CONSULTAR FILA ---
@tool
def consultar_fila():
    """Use para verificar quantos chamados existem na frente do usuário."""
    # Simulação
    return "Existem 4 chamados na fila prioritária. Tempo estimado: 20 min."

#@tool
#def buscar_no_calendario(): 
    """
    Útil para buscar datas, feriados e eventos no calendário acadêmico.
    O input deve ser apenas a pergunta ou termo de busca.
    Exemplo: "início das aulas" ou "feriados novembro".
    """

# --- FERRAMENTA 3: BUSCAR NO PDF (A busca vira uma ferramenta) ---
# (Essa será criada dinamicamente no rag_service, mas saiba que ela existe conceitualmente)