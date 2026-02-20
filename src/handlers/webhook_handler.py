"""
================================================================================
webhook_handler.py — Orquestrador do Fluxo de Atendimento (v2)
================================================================================

CORREÇÕES v2:
  1. processar() agora recebe 'identity' (dict já validado pelo DevGuard),
     não mais o payload bruto do WAHA — elimina extração duplicada de dados
  2. Assinatura do __init__ atualizada para receber menu_service e router_service
     como parâmetros explícitos (facilita testes e evita instanciação interna)
  3. Filtro de mensagem sem texto mantido (mídia sem legenda, stickers, etc.)

FLUXO:
  DevGuard.validar() → identity dict
    → WebhookHandler.processar(identity)
        → MenuService  : decide se é navegação ou ação da LLM
        → RouterService: identifica intenção e monta contexto
        → RagService   : gera resposta com o agente
        → WahaService  : envia resposta ao usuário
================================================================================
"""

import logging
from src.services.menu_service import MenuService
from src.services.router_service import RouterService
from src.services.rag_service import RagService
from src.services.waha_service import WahaService

logger = logging.getLogger(__name__)


class WebhookHandler:
    def __init__(
        self,
        rag_service: RagService,
        waha_service: WahaService,
        menu_service: MenuService,
        router_service: RouterService,
    ):
        self.rag    = rag_service
        self.waha   = waha_service
        self.menu   = menu_service
        self.router = router_service
        logger.info("✅ WebhookHandler inicializado.")

    async def processar(self, identity: dict) -> None:
        """
        Processa uma mensagem já validada e aprovada pelo DevGuard.

        Parâmetro:
          identity : dict retornado pelo DevGuard.validar() com:
            - chat_id      : JID do usuário
            - sender_phone : número sem @
            - body         : texto da mensagem
            - has_media    : bool
            - msg_type     : tipo da mensagem
        """
        chat_id = identity["chat_id"]
        body    = identity["body"]

        # Ignora mensagens sem texto (áudio, figurinha, imagem sem legenda)
        if not body:
            logger.debug("🔇 Mensagem sem texto ignorada para [%s].", chat_id)
            return

        logger.info("📨 [%s] '%s'", chat_id, body[:80])

        # ── MenuService: navegação ou ação da LLM ────────────────────────────
        resultado_menu = self.menu.processar_escolha(chat_id, body)

        # Resposta direta do menu — envia sem passar pela LLM
        if resultado_menu["type"] == "msg":
            await self.waha.enviar_mensagem(chat_id, resultado_menu["content"])
            return

        # Ação da LLM — extrai prompt e contexto
        prompt_base    = resultado_menu.get("prompt", body)
        contexto_extra = resultado_menu.get("contexto_extra", {})
        estado_menu    = contexto_extra.get("estado_menu", "MAIN")

        # ── RouterService: identifica intenção e enriquece o prompt ──────────
        rota           = self.router.analisar(prompt_base, estado_menu=estado_menu)
        ctx_usuario    = self.menu.get_user_context(chat_id)

        prompt_final = self.router.montar_prompt_enriquecido(
            texto_usuario    = prompt_base,
            rota             = rota,
            contexto_usuario = ctx_usuario,
        )

        # Persiste a última intenção identificada
        self.menu.set_user_context(chat_id, {"ultima_intencao": rota["rota"]})

        logger.info("🤖 [%s] rota=%s → RagService", chat_id, rota["rota"])

        # ── RagService: gera resposta ─────────────────────────────────────────
        resposta = self.rag.responder(prompt_final, chat_id)

        # ── WahaService: envia ao usuário ─────────────────────────────────────
        if resposta:
            await self.waha.enviar_mensagem(chat_id, resposta)
        else:
            await self.waha.enviar_mensagem(
                chat_id,
                "Desculpe, não consegui processar sua solicitação. Tente novamente."
            )