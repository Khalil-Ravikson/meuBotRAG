import time
import redis
from fastapi import FastAPI, Request
from src.services.rag_service import RagService
from src.services.waha_service import WahaService
from src.config import settings

app = FastAPI()

rag = RagService()
waha = WahaService()

# --- CONEXÃO REDIS ---
try:
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    print("✅ Conectado ao Redis!")
except:
    print("⚠️ Rodando sem Redis (Rate Limit desativado)")
    r = None

# --- CONFIGURAÇÕES ---
DEV_MODE = True
# 👇 Coloque aqui o número do celular que você usou para testar (o que mandou o "Oi")
DEV_WHITELIST = ["559887680098","175174737518829"] 

@app.on_event("startup")
async def startup_event():
    print(f"🚀 Bot Iniciado! Modo DEV: {DEV_MODE}")
    rag.inicializar()

@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()

        # Validações básicas
        if data.get('event') != 'message': return {"status": "ignored_event"}
        if not data.get('payload'): return {"status": "ignored_empty"}

        payload = data['payload']
        chat_id = payload.get('from')
        event_id = data.get('id') or payload.get('id')
        body = payload.get('body', '').strip()
        has_media = payload.get('hasMedia', False)
        
        # Extrai apenas os números do telefone (ex: 5598988887777)
        sender_phone = chat_id.split('@')[0] if chat_id else "desconhecido"

        # --- 🛡️ 1. FILTROS DE ORIGEM ---
        if payload.get('fromMe'): return {"status": "ignored_self"}
        if "@g.us" in str(chat_id): return {"status": "ignored_group"}
        if "status@broadcast" in str(chat_id): return {"status": "ignored_status"}

        # --- 🛡️ 2. MODO DEV (O Porteiro) ---
        if DEV_MODE and sender_phone not in DEV_WHITELIST:
            print(f"🚧 Modo DEV: Ignorando {sender_phone} (Não está na Whitelist)")
            return {"status": "ignored_dev_mode"}

        # --- 🛡️ 3. FILTRO DE CONTEÚDO (A CORREÇÃO DO LOG VAZIO) ---
        
        # Se for mídia explícita, ignora
        if has_media:
            print(f"🔇 Mídia ignorada de {sender_phone}")
            return {"status": "ignored_media"}

        # 👇 A MÁGICA: Recuperação de Tipo 👇
        # Tenta pegar o tipo. Se vier vazio mas tiver texto, assume que é 'chat'.
        msg_type = payload.get('_data', {}).get('type')
        if not msg_type and body:
            msg_type = 'chat'
        
        # Agora verifica se é um tipo válido
        if msg_type not in ['chat', 'text']:
            print(f"🔇 Tipo ignorado: '{msg_type}'") # Agora vai mostrar o que é, se não for chat
            return {"status": "ignored_msg_type"}

        if not body:
            return {"status": "ignored_empty_body"}

        # --- 🛡️ 4. REDIS (Proteção Anti-Flood) ---
        if r:
            # Deduplicação
            if r.get(f"evt:{event_id}"):
                print(f"♻️ Duplicata ignorada: {event_id}")
                return {"status": "ignored_duplicate"}
            r.setex(f"evt:{event_id}", 300, "1")

            # Rate Limit (5 msgs a cada 10s)
            key = f"rate:{sender_phone}"
            if r.incr(key) == 1: r.expire(key, 10)
            if int(r.get(key) or 0) > 5:
                print(f"🚦 Rate limit estourado: {sender_phone}")
                return {"status": "rate_limited"}

        # --- 🧠 CÉREBRO: Processar e Responder ---
        print(f"🤖 Processando mensagem de {sender_phone}: {body}")
        
        resposta = rag.responder(body, user_id=chat_id)
        
        # Envia a resposta de volta
        waha.enviar_mensagem(chat_id, resposta)

        return {"status": "processed"}

    except Exception as e:
        print(f"❌ Erro no Webhook: {e}")
        return {"status": "error"}