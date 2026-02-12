import smtplib
import os
from email.message import EmailMessage


SMTP_HOST = os.getenv("SMTP_HOST", "host.docker.internal")
SMTP_PORT = int(os.getenv("SMTP_PORT", 1025))


def enviar_email_smtp(
    to: str,
    subject: str,
    body: str,
    from_email: str = "no-reply@uema.dev"
) -> dict:

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as smtp:
            smtp.send_message(msg)

        return {
            "status": "ok",
            "canal": "email",
            "to": to,
            "mensagem": "Email enviado com sucesso"
        }

    except Exception as e:
        return {
            "status": "erro",
            "canal": "email",
            "erro": str(e)
        }
