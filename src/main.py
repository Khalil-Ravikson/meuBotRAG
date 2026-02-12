import redis
import logging
from src.services.logger_service import LogService # Importe no topo
from fastapi import FastAPI, Request
from src.services.rag_service import RagService
from src.services.waha_service import WahaService
from src.config import settings
from src.services.menu_service import MenuService
# --- 0. CONFIGURAÇÃO DE LOGS (SILENCIADOR) ---
# Isso remove o spam de "POST /webhook HTTP/1.1 200 OK" do terminal
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/webhook") == -1

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# --- INICIALIZAÇÃO ---
app = FastAPI()
rag = RagService()
waha = WahaService()
menu = MenuService()
# --- CONEXÃO REDIS (Trava de Segurança) ---
try:
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    r.ping()
    print("✅ Conectado ao Redis!")
except Exception as e:
    print(f"❌ ERRO CRÍTICO: Redis Off. O bot não iniciará para economizar tokens.")
    raise e

# --- CONFIGURAÇÕES ---
DEV_MODE = True
DEV_WHITELIST = ["559887680098", "175174737518829"] 

@app.on_event("startup")
async def startup_event():
    print(f"🚀 Bot Iniciado! Modo DEV: {DEV_MODE}")
    
    # 1. Inicializa a IA e ingestão de dados
    rag.inicializar()
    
    # 2. Configura o Webhook via código
    await waha.configurar_webhook()

@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()

        # 1. Filtro de Evento
        if data.get('event') != 'message': 
            return {"status": "ignored_event"}

        payload = data.get('payload')
        if not payload: 
            return {"status": "ignored_empty"}

        # 2. ANTI-LOOP (Checagem Booleana Estrita)
        # Garante que não responde a si mesmo
        if payload.get('fromMe') is True:
            return {"status": "ignored_self"}

        # 3. Extração de Dados
        chat_id = payload.get('from')
        if not chat_id: return {"status": "no_chat_id"}
        
        sender_phone = chat_id.split('@')[0]
        event_id = data.get('id') or payload.get('id')

        # 4. Filtros de Bloqueio (Grupos, Status, Dev Mode)
        if "@g.us" in str(chat_id): return {"status": "ignored_group"}
        if "status@broadcast" in str(chat_id): return {"status": "ignored_status"}

        if DEV_MODE and sender_phone not in DEV_WHITELIST:
            print(f"🚧 Modo DEV: Ignorando {sender_phone}")
            return {"status": "ignored_dev"}

        # 5. Rate Limit & Deduplicação (Redis)
        # Evita processar a mesma mensagem duas vezes
        if r.get(f"evt:{event_id}"):
            print(f"♻️ Duplicata ignorada: {event_id}")
            return {"status": "ignored_duplicate"}
        r.setex(f"evt:{event_id}", 300, "1")

        # Limite de velocidade (5 msgs a cada 10s)
        key_rate = f"rate:{sender_phone}"
        requests = r.incr(key_rate)
        if requests == 1: r.expire(key_rate, 10)
        
        if requests > 5:
            print(f"🚦 Rate limit: {sender_phone}")
            return {"status": "rate_limited"}

        # 6. Conteúdo da Mensagem
        has_media = payload.get('hasMedia', False)
        raw_body = payload.get('body')
        body = (raw_body or "").strip()

        if has_media:
            print(f"🔇 Mídia ignorada de {sender_phone}")
            return {"status": "ignored_media"}

        if not body:
            return {"status": "ignored_empty_body"}

        # --- 7. CÉREBRO DO ROTEADOR (ROUTER) ---
        # Analisa a intenção antes de chamar a IA cara
        analise = router.analisar(body)
        rota = analise["rota"]
        contexto_extra = analise.get("contexto", "")

        print(f"🧭 Rota: {rota} | Contexto: {contexto_extra}")

# 🚦 LÓGICA DE NAVEGAÇÃO HIERÁRQUICA
        decisao = menu.processar_escolha(sender_phone, body)

        if decisao["type"] == "msg":
            # Responde menus/submenus instantaneamente (Custo 0)
            await waha.enviar_mensagem(chat_id, decisao["content"])
            return {"status": "menu_ok"}

        # Se for action, envia para a IA com o prompt que o menu preparou
        prompt_final = decisao["prompt"]
        resposta = rag.responder(prompt_final, user_id=sender_phone)
        await waha.enviar_mensagem(chat_id, resposta)
        
        return {"status": "processed"}

    except Exception as e:
        # O teu LogService entra em ação aqui
        logger_service.log_error(sender_phone, "WEBHOOK_CRITICAL", str(e))
        await waha.enviar_mensagem(chat_id, "Estou com uma instabilidade momentânea. Tente em 1 minuto.")
        return {"status": "error_handled"}
    
    except Exception as e:
        # 1. Loga o erro detalhado no Redis
        logger = LogService()
        logger.log_error(sender_phone, "CRITICAL_WEBHOOK_FAILURE", str(e))
        
        print(f"❌ Erro Crítico Controlado: {e}")
        
        # 2. Resposta de Emergência para o Usuário (Self-Healing)
        # Se a IA morreu, o Python assume e manda um aviso.
        msg_erro = "Indisponibilidade momentânea nos meus sistemas neurais. 😵‍💫\nPor favor, tente novamente em 1 minuto."
        
        # Tenta enviar o aviso pelo Waha (se o Waha estiver vivo)
        try:
            await waha.enviar_mensagem(chat_id, msg_erro)
        except:
            print("💀 Waha também morreu. Nada a fazer.")
            
        return {"status": "error_handled"}