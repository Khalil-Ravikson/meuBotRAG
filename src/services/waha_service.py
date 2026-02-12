import httpx
import logging
import asyncio
from src.config import settings

logger = logging.getLogger(__name__)

class WahaService:
    def __init__(self):
        self.base_url = settings.WAHA_BASE_URL
        self.api_key = settings.WAHA_API_KEY
        self.headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
        }
        # Se não tiver URL no env, assume o padrão interno do Docker
        self.webhook_url = settings.WHATSAPP_HOOK_URL or "http://bot-rag:8000/webhook"
        self.events = ["message"]

    async def configurar_webhook(self):
        """
        Força o registro do Webhook no Waha via API assim que o bot liga.
        """
        url = f"{self.base_url}/api/sessions/default/webhook"
        payload = {
            "url": self.webhook_url,
            "events": self.events
        }

        print(f"🔌 Tentando registrar Webhook em: {self.webhook_url}...")

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # Tenta registrar (POST)
                r = await client.post(url, json=payload, headers=self.headers)
                
                if r.status_code in [200, 201]:
                    print(f"✅ Webhook configurado com sucesso!")
                else:
                    # Se der erro, tenta a rota alternativa (PUT/PATCH) dependendo da versão
                    print(f"⚠️ Aviso Waha ({r.status_code}): {r.text}")

            except Exception as e:
                print(f"❌ Falha ao configurar Webhook: {e}")

    async def enviar_mensagem(self, chat_id: str, texto: str):
        if not chat_id or not texto:
            return

        url = f"{self.base_url}/api/sendText"
        payload = {
            "session": "default",
            "chatId": chat_id,
            "text": texto
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.post(url, json=payload, headers=self.headers)
                if response.status_code not in [200, 201]:
                    print(f"⚠️ Erro envio Waha ({response.status_code}): {response.text}")
            except Exception as e:
                print(f"❌ Erro de Conexão: {e}")