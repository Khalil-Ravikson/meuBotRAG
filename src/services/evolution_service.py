"""
services/evolution_service.py — v2 (retry @lid + envio robusto)
================================================================

NOVO em v2:
  enviar_mensagem() tenta dois formatos quando o primeiro falha com 400:
    1. chat_id como veio do dev_guard (pode ser @lid ou @s.whatsapp.net)
    2. Apenas os dígitos + @s.whatsapp.net (fallback para @lid)

  Isto resolve o "exists:false" sem depender do senderPn.
"""
from __future__ import annotations
import logging
import httpx

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)


class EvolutionService:
    def __init__(self):
        self.base_url = settings.EVOLUTION_BASE_URL.rstrip("/")
        self.api_key  = settings.EVOLUTION_API_KEY
        self.instance = settings.EVOLUTION_INSTANCE_NAME
        self.headers  = {
            "Content-Type": "application/json",
            "apikey":       self.api_key,
        }
        self.webhook_url = settings.WHATSAPP_HOOK_URL

    # ------------------------------------------------------------------
    # STATUS E AUTO-RECUPERAÇÃO
    # ------------------------------------------------------------------

    async def verificar_instancia(self) -> str | None:
        url = f"{self.base_url}/instance/connectionState/{self.instance}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r     = await client.get(url, headers=self.headers)
                if r.status_code == 200:
                    estado = r.json().get("instance", {}).get("state", "UNKNOWN")
                    logger.info("ℹ️  Evolution Instância '%s': %s", self.instance, estado)
                    return estado
                elif r.status_code == 404:
                    return "NOT_FOUND"
                logger.warning("⚠️  Status Evolution: %s | %s", r.status_code, r.text[:200])
                return None
            except Exception as e:
                logger.error("❌ Erro ao verificar Evolution API: %s", e)
                return None

    async def criar_instancia(self) -> None:
        url     = f"{self.base_url}/instance/create"
        payload = {
            "instanceName": self.instance,
            "qrcode":       True,
            "integration":  "WHATSAPP-BAILEYS",
        }
        logger.info("⚙️  Criando instância '%s'...", self.instance)
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers)
                if r.status_code in (200, 201):
                    logger.info("✅ Instância criada!")
                else:
                    logger.warning("⚠️  Falha ao criar: %s | %s", r.status_code, r.text[:200])
            except Exception as e:
                logger.exception("❌ Erro ao criar instância: %s", e)

    async def configurar_webhook(self) -> None:
        url     = f"{self.base_url}/webhook/set/{self.instance}"
        payload = {
            "webhook": {
                "enabled":        True,
                "url":            self.webhook_url,
                "webhookByEvents": False,
                "events":         ["MESSAGES_UPSERT", "CONNECTION_UPDATE"],
            }
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers)
                if r.status_code in (200, 201):
                    logger.info("✅ Webhook configurado → %s", self.webhook_url)
                else:
                    logger.warning("⚠️  Webhook falhou: %s | %s", r.status_code, r.text[:200])
            except Exception as e:
                logger.error("❌ Erro ao configurar webhook: %s", e)

    async def inicializar(self) -> None:
        logger.info("🚀 Inicializando EvolutionService...")
        status = await self.verificar_instancia()
        if status == "NOT_FOUND":
            logger.info("▶️  Instância não existe. Criando...")
            await self.criar_instancia()
        if status is not None:
            await self.configurar_webhook()

    # ------------------------------------------------------------------
    # ENVIO DE MENSAGENS — com retry automático para @lid
    # ------------------------------------------------------------------

    async def enviar_mensagem(self, chat_id: str, texto: str) -> bool:
        """
        Envia mensagem de texto com retry automático.

        Estratégia para @lid:
          Tentativa 1: envia com o chat_id como veio (pode ser @lid)
          Tentativa 2: se falhar com 400, converte para @s.whatsapp.net

        Retorna True se enviou com sucesso, False caso contrário.
        """
        if not chat_id or not texto:
            logger.warning("⚠️  enviar_mensagem: chat_id ou texto vazio.")
            return False

        url = f"{self.base_url}/message/sendText/{self.instance}"

        # Monta lista de candidatos a tentar
        candidatos = [chat_id]

        # Se for @lid, adiciona fallback com @s.whatsapp.net
        if "@lid" in chat_id:
            digitos = "".join(filter(str.isdigit, chat_id.split("@")[0]))
            if len(digitos) >= 10:
                fallback = f"{digitos}@s.whatsapp.net"
                candidatos.append(fallback)
                logger.debug("📱 @lid detectado — tentará também: %s", fallback)

        async with httpx.AsyncClient(timeout=15.0) as client:
            for i, numero in enumerate(candidatos, 1):
                payload = {
                    "number": numero,
                    "text":   texto,
                    "delay":  1200,
                }
                try:
                    r = await client.post(url, json=payload, headers=self.headers)
                    if r.status_code in (200, 201):
                        logger.info("✅ Mensagem enviada → %s", numero)
                        return True
                    elif r.status_code == 400 and i < len(candidatos):
                        logger.debug(
                            "⚠️  Tentativa %d falhou (400) para %s — tentando formato alternativo.",
                            i, numero,
                        )
                        continue
                    else:
                        logger.warning(
                            "⚠️  Falha ao enviar para %s. Status %s | %s",
                            numero, r.status_code, r.text[:200],
                        )
                except httpx.ConnectError:
                    logger.error("❌ Não consegue conectar à Evolution API: %s", self.base_url)
                    return False
                except httpx.TimeoutException:
                    logger.error("❌ Timeout ao enviar para %s", numero)
                    return False
                except Exception as e:
                    logger.exception("❌ Erro inesperado ao enviar: %s", e)
                    return False

        return False