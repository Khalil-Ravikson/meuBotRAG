import requests
from src.config import settings

class WahaService:
    def __init__(self):
        self.base_url = settings.WAHA_BASE_URL
        self.headers = {
            "X-Api-Key": settings.WAHA_API_KEY,
            "Content-Type": "application/json"
        }

    def enviar_mensagem(self, chat_id: str, texto: str):
        url = f"{self.base_url}/api/sendText"
        payload = {
            "session": "default",
            "chatId": chat_id,
            "text": texto
        }
        try:
            print(f"📤 Enviando para {chat_id} via {url}...")
            r = requests.post(url, json=payload, headers=self.headers)
            if r.status_code == 201:
                print("✅ Mensagem enviada!")
            else:
                print(f"⚠️ Erro WAHA: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"❌ Erro de conexão com WAHA: {e}")