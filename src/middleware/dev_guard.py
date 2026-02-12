
# Função em Andamento:
from fastapi import Request
from fastapi.responses import JSONResponse
from src.config import settings

DEV_MODE = True  # 🔥 MODO DEV ATIVO

async def dev_guard_middleware(request: Request, call_next):
    # Só intercepta o webhook
    if request.url.path != "/webhook":
        return await call_next(request)

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"status": "invalid_json"}, status_code=400)

    payload = data.get("payload", {})
    chat_id = payload.get("from", "")

    # 🔁 Anti-loop
    if payload.get("fromMe"):
        return JSONResponse({"status": "ignored_self"})

    # 🚫 Grupo
    if "@g.us" in chat_id:
        return JSONResponse({"status": "ignored_group"})

    # 🚫 Canal
    if "@newsletter" in chat_id:
        return JSONResponse({"status": "ignored_newsletter"})

    # 🚫 Broadcast
    if "status@broadcast" in chat_id:
        return JSONResponse({"status": "ignored_broadcast"})

    # 🔐 DEV MODE: só sessão default
    if DEV_MODE and payload.get("session") != "default":
        return JSONResponse({"status": "ignored_other_session"})

    # Passou pelo porteiro → segue pra rota
    return await call_next(request)
