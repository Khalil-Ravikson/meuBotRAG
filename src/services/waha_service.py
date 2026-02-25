"""
src/services/waha_service.py — Integração com WAHA (WhatsApp HTTP API)
===================================================================
Atualizado com lógica de Auto-Recuperação (Self-Healing) e criação
automática de sessão com configuração embutida do Webhook.
"""
from __future__ import annotations
import logging
import httpx

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

class WahaService:
    def __init__(self):
        self.base_url    = settings.WAHA_BASE_URL.rstrip("/")
        self.api_key     = settings.WAHA_API_KEY
        self.session     = settings.WAHA_SESSION
        self.headers     = {
            "Content-Type": "application/json",
            "X-Api-Key":    self.api_key,
        }
        self.webhook_url = settings.WHATSAPP_HOOK_URL
        self.events      = ["message", "session.status"]

    # ------------------------------------------------------------------
    # GERENCIAMENTO DE SESSÃO (AUTO-RECUPERAÇÃO)
    # ------------------------------------------------------------------

    async def verificar_sessao(self) -> str | None:
        """Consulta o status atual da sessão. Retorna None se o WAHA estiver offline."""
        url = f"{self.base_url}/api/sessions/{self.session}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.get(url, headers=self.headers)
                if r.status_code == 200:
                    status = r.json().get("status", "UNKNOWN")
                    logger.info("ℹ️  Sessão '%s': %s", self.session, status)
                    return status
                elif r.status_code == 404:
                    # Sessão não existe
                    return "NOT_FOUND"
                logger.warning("⚠️  Status sessão: %s | %s", r.status_code, r.text)
                return None
            except httpx.ConnectError:
                logger.error("❌ WAHA inacessível: %s", self.base_url)
            except httpx.TimeoutException:
                logger.error("❌ Timeout ao verificar sessão.")
            except Exception as e:
                logger.exception("❌ Erro ao verificar sessão: %s", e)
            return None

    async def deletar_sessao(self) -> bool:
        """Deleta a sessão atual (necessário quando ela corrompe e fica FAILED)."""
        url = f"{self.base_url}/api/sessions/{self.session}"
        logger.warning("🗑️  Deletando sessão corrompida/parada '%s'...", self.session)
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                r = await client.delete(url, headers=self.headers)
                if r.status_code in (200, 204, 404):
                    logger.info("✅ Sessão deletada (ou já não existia).")
                    return True
                logger.error("❌ Falha ao deletar sessão: %s | %s", r.status_code, r.text)
            except Exception as e:
                logger.exception("❌ Erro ao deletar sessão: %s", e)
        return False

    async def criar_sessao(self) -> None:
        """
        Cria (ou inicia) a sessão já embutindo a configuração do Webhook.
        Isso mata a necessidade de chamar o endpoint de webhook separadamente na criação.
        """
        url = f"{self.base_url}/api/sessions"
        
        # Payload completo usando as configurações recomendadas pela documentação
        payload = {
            "name": self.session,
            "config": {
                "webhooks": [{
                    "url": self.webhook_url,
                    "events": self.events,
                }],
                # Metadados ajudam a identificar os webhooks depois
                "metadata": {
                    "app": "bot_rag",
                    "ambiente": "dev" if getattr(settings, "DEV_MODE", False) else "prod"
                }
            }
        }

        logger.info("⚙️  Criando/Iniciando sessão '%s' com Webhook embutido...", self.session)
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers)
                if r.status_code in (200, 201):
                    logger.info("✅ Sessão '%s' criada e iniciando!", self.session)
                else:
                    logger.warning("⚠️  Falha ao criar sessão %s | %s", r.status_code, r.text)
            except Exception as e:
                logger.exception("❌ Erro ao criar sessão: %s", e)

    # ------------------------------------------------------------------
    # INICIALIZAÇÃO (Chamada no startup do main.py)
    # ------------------------------------------------------------------

    async def inicializar(self) -> None:
        """
        Fluxo principal de auto-healing. Avalia o status e toma a decisão.
        """
        logger.info("🚀 Inicializando WahaService (Auto-Healing ativado)...")
        status = await self.verificar_sessao()

        if status is None:
            logger.error("❌ WAHA inacessível. O container do WAHA está rodando?")
            return

        # 1. Se a sessão falhou, travou, ou não existe, vamos limpar e recriar
        if status in ("FAILED", "STOPPED", "NOT_FOUND"):
            if status == "FAILED":
                logger.error("🚨 Sessão corrompida (FAILED) detectada! Executando hard-reset...")
                await self.deletar_sessao()
            
            # Cria a sessão do zero (já com o webhook configurado)
            await self.criar_sessao()
            
        # 2. Se ela já estava viva, apenas garantimos que o webhook aponta pro lugar certo
        elif status in ("WORKING", "SCAN_QR_CODE", "STARTING"):
            logger.info("👍 Sessão operante (Status: %s). Atualizando webhook por segurança...", status)
            await self.configurar_webhook()

    # ------------------------------------------------------------------
    # FALLBACK DE WEBHOOK E ENVIO DE MENSAGENS
    # ------------------------------------------------------------------

    async def configurar_webhook(self) -> None:
        """Atualiza APENAS o webhook via PUT (útil para quando a sessão já existe)."""
        url = f"{self.base_url}/api/sessions/{self.session}"
        payload = {
            "name": self.session,
            "config": {
                "webhooks": [{
                    "url": self.webhook_url,
                    "events": self.events,
                }]
            },
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.put(url, json=payload, headers=self.headers)
                if r.status_code == 200:
                    logger.info("✅ Webhook atualizado → %s", self.webhook_url)
            except Exception as e:
                logger.error("❌ Erro ao atualizar webhook: %s", e)

    async def enviar_mensagem(self, chat_id: str, texto: str) -> None:
        """Envia mensagem de texto via POST /api/sendText."""
        if not chat_id or not texto:
            logger.warning("⚠️  enviar_mensagem: chat_id ou texto vazio.")
            return

        url = f"{self.base_url}/api/sendText"
        payload = {
            "session": self.session,
            "chatId":  chat_id,
            "text":    texto,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers)
                if r.status_code in (200, 201):
                    logger.info("✅ Mensagem enviada para %s", chat_id)
                else:
                    logger.warning("⚠️  Falha ao enviar. Status %s | %s", r.status_code, r.text)
            except httpx.ConnectError:
                logger.error("❌ Não foi possível conectar ao WAHA: %s", self.base_url)
            except httpx.TimeoutException:
                logger.error("❌ Timeout ao enviar para %s", chat_id)
            except Exception as e:
                logger.exception("❌ Erro inesperado ao enviar mensagem: %s", e)