import redis
import logging

from fastapi import FastAPI, Request
from src.config import settings
from src.services.rag_service import RagService
from src.services.waha_service import WahaService
from src.services.menu_service import MenuService
from src.services.router_service import RouterService
from src.services.logger_service import LogService
from src.services.redis_history import limpar_historico

# --- SILENCIADOR DE LOG DO WEBHOOK ---
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/webhook" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# --- INICIALIZAÇÃO ---
app    = FastAPI()
rag    = RagService()
waha   = WahaService()
menu   = MenuService()
router = RouterService()
logger = LogService()

# --- REDIS ---
try:
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    r.ping()
    print("✅ Redis conectado!")
except Exception as e:
    print("❌ ERRO CRÍTICO: Redis offline. Bot não iniciará.")
    raise e

# --- CONFIG ---
DEV_MODE      = True
DEV_WHITELIST = {"559887680098", "175174737518829"}  # set é O(1) no lookup


# =============================================================================
# STARTUP
# =============================================================================
@app.on_event("startup")
async def startup_event():
    print(f"🚀 Bot iniciado! Modo DEV: {DEV_MODE}")
    rag.inicializar()
    await waha.configurar_webhook()


# =============================================================================
# WEBHOOK
# =============================================================================
@app.post("/webhook")
async def webhook(request: Request):
    chat_id      = None
    sender_phone = None

    try:
        data = await request.json()

        # 1. Filtro de evento
        if data.get("event") != "message":
            return {"status": "ignored_event"}

        payload = data.get("payload")
        if not payload or payload.get("fromMe") is True:
            return {"status": "ignored_self"}

        # 2. Extração
        chat_id      = payload.get("from", "")
        sender_phone = chat_id.split("@")[0]
        event_id     = data.get("id") or payload.get("id")
        body         = (payload.get("body") or "").strip()

        # 3. Bloqueios
        if "@g.us" in chat_id or "status@broadcast" in chat_id:
            return {"status": "ignored_group"}

        if DEV_MODE and sender_phone not in DEV_WHITELIST:
            return {"status": "ignored_dev"}

        # 4. Deduplicação
        if r.get(f"evt:{event_id}"):
            return {"status": "ignored_duplicate"}
        r.setex(f"evt:{event_id}", 300, "1")

        # 5. Rate limit — 5 msgs / 10s por número
        key_rate = f"rate:{sender_phone}"
        count = r.incr(key_rate)
        if count == 1:
            r.expire(key_rate, 10)
        if count > 5:
            return {"status": "rate_limited"}

        # 6. Ignora vazio e mídia
        if not body or payload.get("hasMedia", False):
            return {"status": "ignored_content"}

        # =====================================================================
        # 🚦 ROTEAMENTO HIERÁRQUICO
        #
        # Ordem de decisão:
        #   MenuService  →  resposta fixa de menu (zero tokens)
        #   RouterService → enriquece prompt com contexto de rota
        #   RagService   →  chama a IA com prompt enriquecido
        # =====================================================================

        estado_atual = menu.get_user_state(sender_phone)
        decisao      = menu.processar_escolha(sender_phone, body)

        # --- Resposta de menu fixo (sem IA) ---
        if decisao["type"] == "msg":
            await waha.enviar_mensagem(chat_id, decisao["content"])
            return {"status": "menu_ok"}

        # --- Ação: enriquece o prompt via RouterService ---
        prompt_base = decisao["prompt"]
        rota        = router.analisar(prompt_base, estado_menu=estado_atual)

        # RESET: limpa histórico Redis e exibe menu
        if rota["rota"] == "RESET":
            limpar_historico(sender_phone)
            menu.clear_user_state(sender_phone)
            await waha.enviar_mensagem(chat_id, menu.menus["MAIN"]["msg"])
            return {"status": "reset_ok"}

        # Adiciona contexto de rota como instrução invisível para a IA
        if rota["rota"] != "GERAL":
            prompt_final = f"[CONTEXTO: {rota['contexto']}]\n{prompt_base}"
        else:
            prompt_final = prompt_base

        print(f"🤖 [{rota['rota']}] {sender_phone}: {prompt_base[:60]}...")

        # --- Chama a IA ---
        resposta = rag.responder(prompt_final, user_id=sender_phone)
        await waha.enviar_mensagem(chat_id, resposta)

        return {"status": "processed"}

    except Exception as e:
        erro_str = str(e)
        phone_log = sender_phone or "unknown"
        print(f"❌ Erro crítico [{phone_log}]: {erro_str}")
        logger.log_error(phone_log, "WEBHOOK_CRITICAL", erro_str)

        # Tenta avisar o usuário — silencia se waha também falhar
        if chat_id:
            try:
                await waha.enviar_mensagem(
                    chat_id,
                    "Estou com uma instabilidade momentânea. Tente novamente em 1 minuto. 🙏"
                )
            except Exception:
                pass

        return {"status": "error_handled"}