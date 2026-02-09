import requests
import time
from src.config import settings

class WahaService:
    def __init__(self):
        self.base_url = settings.WAHA_BASE_URL
        self.headers = {
            "X-Api-Key": settings.WAHA_API_KEY,
            "Content-Type": "application/json"
        }
        # 🚀 Já configura o Webhook assim que a classe é iniciada
        self.configurar_webhook()

    def configurar_webhook(self):
        """
        Diz ao WAHA: "Mande todas as mensagens para http://bot-rag:8000/webhook"
        """
        print("🔗 Configurando Webhook no WAHA...")
        
        url = f"{self.base_url}/api/sessions/default/webhook"
        
        # Endereço interno do Docker (O WAHA chama o Bot por aqui)
        webhook_target = "http://bot-rag:8000/webhook"
        
        payload = {
            "url": webhook_target,
            "events": ["message", "session.status"], # Escuta mensagens e status da conexão
            "allUnreadOnStart": False # Evita processar mensagens velhas ao reiniciar
        }

        try:
            # Tenta configurar. Se o WAHA estiver acordando, tenta algumas vezes.
            for tentativa in range(3):
                try:
                    r = requests.post(url, json=payload, headers=self.headers)
                    if r.status_code in [200, 201]:
                        print(f"✅ Webhook configurado com sucesso: {webhook_target}")
                        return
                    else:
                        print(f"⚠️ Tentativa {tentativa+1}: WAHA retornou {r.status_code} - {r.text}")
                except requests.exceptions.ConnectionError:
                    print(f"⏳ Tentativa {tentativa+1}: WAHA ainda não está acessível...")
                
                time.sleep(2) # Espera 2 segundos antes de tentar de novo
            
            print("❌ Falha ao configurar Webhook após tentativas.")

        except Exception as e:
            print(f"❌ Erro crítico ao configurar webhook: {e}")

    def enviar_mensagem(self, chat_id: str, texto: str):
        url = f"{self.base_url}/api/sendText"
        payload = {
            "session": "default",
            "chatId": chat_id,
            "text": texto
        }
        try:
            print(f"📤 Enviando para {chat_id}...")
            r = requests.post(url, json=payload, headers=self.headers)
            if r.status_code == 201:
                print("✅ Mensagem enviada!")
            else:
                print(f"⚠️ Erro WAHA: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"❌ Erro de conexão com WAHA: {e}")