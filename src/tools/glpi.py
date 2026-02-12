from langchain_core.tools import tool
import random
import json
@tool
def abrir_chamado_glpi(titulo: str, descricao: str, local: str = "Não informado", urgencia: str = "media"):
    """
    Use esta ferramenta para abrir chamados de suporte técnico (GLPI).
    Deve ser usada quando o usuário relatar problemas com computadores, internet, impressoras ou sistemas.

    Args:
        titulo: Um resumo curto do problema (ex: "Sem internet no laboratório").
        descricao: Detalhes completos do relato do usuário.
        local: Onde o problema está ocorrendo (Sala, Bloco, Laboratório).
        urgencia: Nível de prioridade. Pode ser 'baixa', 'media' ou 'alta'.
    """
    try:
        # --- LÓGICA DE MOCK (SIMULAÇÃO) ---
        # Aqui fingimos que enviamos para a API do GLPI
        fake_id = random.randint(5000, 9999)
        
        print(f"🛠️ [GLPI MOCK] Criando Ticket...")
        print(f"   | Título: {titulo}")
        print(f"   | Local: {local}")
        print(f"   | Urgência: {urgencia}")

        resultado = {
            "status": "sucesso",
            "action": "abrir_chamado_glpi",
            "message": f"Chamado aberto com sucesso sob o protocolo #{fake_id}.",
            "data": {
                "id": fake_id,
                "titulo": titulo,
                "status": "novo",
                "link": f"https://glpi.uema.br/front/ticket.form.php?id={fake_id}"
            }
        }
        # RETORNA COMO STRING JSON (Isso resolve o erro 'NoneType is not iterable')
        return json.dumps(resultado, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"status": "erro", "error": str(e)})