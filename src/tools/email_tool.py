from langchain_core.tools import tool
from src.services.email_service import enviar_email_smtp
import json  # <--- IMPORTANTE

@tool
def enviar_email_notificacao(destinatario: str, assunto: str, mensagem: str):
    """
    Envia email institucional.
    Use para enviar notificações ou confirmações por email.
    Args:
        destinatario: O endereço de email (ex: usuario@email.com).
        assunto: O título do email.
        mensagem: O corpo do texto do email.
    """
    try:
        enviar_email_smtp(
            to=destinatario,
            subject=assunto,
            body=mensagem
        )

        # Monta o resultado
        resultado = {
            "status": "ok",
            "action": "enviar_email_notificacao",
            "message": f"Email enviado com sucesso para {destinatario}.",
            "data": {
                "destinatario": destinatario,
                "assunto": assunto
            }
        }
        
        # RETORNA STRING JSON (Blinda contra erro de NoneType no Agente)
        return json.dumps(resultado, ensure_ascii=False)

    except Exception as e:
        erro_res = {
            "status": "erro",
            "action": "enviar_email_notificacao",
            "message": "Falha técnica ao enviar o email.",
            "error": str(e)
        }
        return json.dumps(erro_res, ensure_ascii=False)