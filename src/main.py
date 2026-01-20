from fastapi import FastAPI, Request, BackgroundTasks
from src.services.waha_service import WahaService
from src.services.rag_service import RagService

# Instancia os serviços
app = FastAPI(title="Bot Modular RAG")
waha = WahaService()
rag = RagService()

@app.on_event("startup")
async def startup_event():
    """Roda quando o servidor liga"""
    rag.inicializar()
    # OBS: Descomente a linha abaixo apenas na primeira vez para carregar o PDF
    # Ou crie uma lógica para verificar se o banco está vazio
    rag.ingerir_pdf() 

def processar_background(chat_id: str, texto: str, sender_name: str):
    """Tarefa em segundo plano"""
    print(f"🧠 Processando para {sender_name}: {texto}")
    
    # 1. Pega resposta da IA
    resposta = rag.responder(texto)
    
    # 2. Envia volta
    waha.enviar_mensagem(chat_id, f"🤖 {resposta}")

@app.post("/webhook")
async def webhook(req: Request, background_tasks: BackgroundTasks):
    try:
        data = await req.json()
        # O WAHA às vezes manda o payload direto ou dentro de 'payload'
        payload = data.get('payload', data)

        # Extrai os dados com segurança (.get evita quebrar se não existir)
        chat_id = payload.get('from')
        texto = payload.get('body')
        sender_name = payload.get('pushName', 'Usuário')

        # --- 🚫 FILTRO 1: SEGURANÇA (Evita crash com imagem/figurinha) ---
        # Se não tiver texto ou não for string, ignora.
        if not texto or not isinstance(texto, str):
            # print(f"🔇 Mensagem sem texto ignorada.")
            return {"status": "ignored_empty"}

        # --- 🚫 FILTRO 2: IGNORAR GRUPOS ---
        # Se o ID terminar em @g.us, é grupo. O bot fica quieto.
        if "@g.us" in str(chat_id):
            print(f"🔇 Mensagem de Grupo ignorada: {sender_name}")
            return {"status": "ignored_group"}

        # --- 🚫 FILTRO 3: IGNORAR A SI MESMO ---
        if payload.get('fromMe', False):
            return {"status": "ignored_self"}

        # --- ✅ PASSOU NOS FILTROS? PROCESSA! ---
        print(f"📩 Recebido de {sender_name}: {texto}")
        
        # Agenda o processamento
        background_tasks.add_task(processar_background, chat_id, texto, sender_name)
            
    except Exception as e:
        print(f"❌ Erro no webhook: {e}")
        
    return {"status": "ok"}