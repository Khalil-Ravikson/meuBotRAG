"""
src/services/evolution_service.py — Integração com Evolution API v2
===================================================================
Gerencia a criação de instâncias, webhooks e envio de mensagens.
"""
from __future__ import annotations
import logging
import httpx

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

class EvolutionService:
    def __init__(self):
        self.base_url    = settings.EVOLUTION_BASE_URL.rstrip("/")
        self.api_key     = settings.EVOLUTION_API_KEY
        self.instance    = settings.EVOLUTION_INSTANCE_NAME
        self.headers     = {
            "Content-Type": "application/json",
            "apikey":       self.api_key,
        }
        self.webhook_url = settings.WHATSAPP_HOOK_URL

    # ------------------------------------------------------------------
    # STATUS E AUTO-RECUPERAÇÃO
    # ------------------------------------------------------------------

    async def verificar_instancia(self) -> str | None:
        """Verifica o status da conexão da instância."""
        url = f"{self.base_url}/instance/connectionState/{self.instance}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.get(url, headers=self.headers)
                if r.status_code == 200:
                    dados = r.json()
                    # A Evolution v2 retorna o estado da instância dentro de 'instance'
                    estado = dados.get("instance", {}).get("state", "UNKNOWN")
                    logger.info("ℹ️  Evolution Instância '%s': %s", self.instance, estado)
                    return estado
                elif r.status_code == 404:
                    return "NOT_FOUND"
                logger.warning("⚠️  Status Evolution: %s | %s", r.status_code, r.text)
                return None
            except Exception as e:
                logger.error("❌ Erro ao verificar Evolution API: %s", e)
                return None

    async def criar_instancia(self) -> None:
        """Cria a instância no Evolution API."""
        url = f"{self.base_url}/instance/create"
        payload = {
            "instanceName": self.instance,
            "qrcode": True, # Gera QR Code para leitura
            "integration": "WHATSAPP-BAILEYS"
        }
        logger.info("⚙️  Criando instância '%s' na Evolution API...", self.instance)
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers)
                if r.status_code in (200, 201):
                    logger.info("✅ Instância criada com sucesso!")
                else:
                    logger.warning("⚠️  Falha ao criar instância: %s | %s", r.status_code, r.text)
            except Exception as e:
                logger.exception("❌ Erro ao criar instância: %s", e)

    async def configurar_webhook(self) -> None:
        """Seta o Webhook global para a instância."""
        url = f"{self.base_url}/webhook/set/{self.instance}"
        payload = {
            "webhook": {
                "enabled": True,
                "url": self.webhook_url,
                "webhookByEvents": False,
                "events": [
                    "MESSAGES_UPSERT",
                    "CONNECTION_UPDATE"
                ]
            }
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers)
                if r.status_code in (200, 201):
                    logger.info("✅ Webhook Evolution configurado → %s", self.webhook_url)
                else:
                    logger.warning("⚠️  Falha no Webhook: %s | %s", r.status_code, r.text)
            except Exception as e:
                logger.error("❌ Erro ao configurar webhook: %s", e)

    async def inicializar(self) -> None:
        """Chamado no startup do main.py."""
        logger.info("🚀 Inicializando EvolutionService...")
        status = await self.verificar_instancia()

        if status == "NOT_FOUND":
            logger.info("▶️  Instância não existe. Criando nova...")
            await self.criar_instancia()
        
        # Garante que o webhook está sempre apontando pro lugar certo
        if status is not None:
            await self.configurar_webhook()

    # ------------------------------------------------------------------
    # ENVIO DE MENSAGENS
    # ------------------------------------------------------------------

    async def enviar_mensagem(self, chat_id: str, texto: str) -> None:
        """Envia mensagem de texto."""
        if not chat_id or not texto:
            return

        # A Evolution v2 usa /message/sendText/{instanceName}
        url = f"{self.base_url}/message/sendText/{self.instance}"
        
        # chat_id costuma ser apenas os números, mas a Evolution aceita o sufixo @s.whatsapp.net
        payload = {
            "number": chat_id, 
            "text": texto,
            "delay": 1200 # Digitando por 1.2 segundos (humanizado)
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers)
                if r.status_code in (200, 201):
                    logger.info("✅ Mensagem enviada para %s", chat_id)
                else:
                    logger.warning("⚠️  Falha ao enviar. Status %s | %s", r.status_code, r.text)
            except Exception as e:
                logger.exception("❌ Erro inesperado ao enviar mensagem: %s", e)