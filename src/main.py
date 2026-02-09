import requests
from fastapi import FastAPI, Request

from src.services.rag_service import RagService
from src.config import settings
from src.middleware.dev_guard import dev_guard_middleware  # 👈 middleware

app = FastAPI()
rag = RagService()

# 🔐 REGISTRA O MIDDLEWARE (PORTEIRO)
app.middleware("http")(dev_guard_middleware)


@app.on_event("startup")
async def startup_event():
    print("🚀 Iniciando Bot (Modo DEV — só eu converso)")
    rag.inicializar()
    rag.ingerir_pdf()


@app.post("/webhook")
async def webhook(request: Request):
    """
    ⚠️ ATENÇÃO:
    - Todos os filtros (anti-loop, grupo, canal, etc.)
      já rodam ANTES aqui no middleware.
    - Aqui só entra mensagem válida.
    """
    try:
        data = await request.json()
        payload = data.get("payload", {})
        chat_id = payload.get("from")

        texto_usuario = ""

        # 🎤 ÁUDIO
        if (
            payload.get("hasMedia")
            and payload.get("media", {})
            .get("mimetype", "")
            .startswith("audio")
        ):
            print(f"🎤 Áudio detectado de {chat_id}")

            media_url = payload["media"]["url"]
            if not media_url.startswith("http"):
                media_url = f"{settings.WAHA_BASE_URL}{media_url}"

            try:
                content = requests.get(media_url).content
                temp_filename = f"/tmp/{chat_id}.ogg"

                with open(temp_filename, "wb") as f:
                    f.write(content)

                texto_usuario = rag.transcrever_audio(temp_filename)
                print(f"📝 Transcrição: {texto_usuario}")

            except Exception as e:
                print(f"❌ Erro ao processar áudio: {e}")
                return {"status": "audio_error"}

        # 💬 TEXTO
        else:
            texto_usuario = payload.get("body", "")

        if not texto_usuario:
            print("⚠️ Mensagem vazia")
            return {"status": "empty"}

        # 🤖 IA RESPONDE
        print(f"🤖 Agente pensando para {chat_id}...")
        resposta = rag.responder(texto_usuario, user_id=chat_id)
        print(f"📤 Resposta IA: {resposta}")

        # 📡 ENVIA VIA WAHA
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": settings.WAHA_API_KEY,
        }

        payload_resp = {
            "chatId": chat_id,
            "text": resposta,
            "session": "default",
        }

        r = requests.post(
            f"{settings.WAHA_BASE_URL}/api/sendText",
            json=payload_resp,
            headers=headers,
        )

        print(f"✅ Enviado para WAHA: {r.status_code}")
        return {"status": "sent"}

    except Exception as e:
        print(f"❌ ERRO NO WEBHOOK: {e}")
        return {"status": "error"}
