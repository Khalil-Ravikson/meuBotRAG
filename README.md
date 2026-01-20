# 🤖 MeuBotRAG - Assistente de WhatsApp com IA

Este projeto é um Chatbot inteligente para WhatsApp que utiliza **RAG (Retrieval-Augmented Generation)**. Ele é capaz de ler documentos PDF (como receitas), armazenar o conhecimento e responder perguntas dos usuários de forma contextualizada usando IA Generativa.

## 🚀 Tecnologias Utilizadas

- **Python 3.10** (Backend com FastAPI)
- **Docker & Docker Compose** (Containerização completa)
- **PostgreSQL + PgVector** (Banco de dados vetorial para memória da IA)
- **LangChain** (Orquestração da IA)
- **Groq API** (Llama 3 para geração de respostas rápidas)
- **WAHA (Whatsapp HTTP API)** (Integração com WhatsApp)

## 📂 Estrutura do Projeto

```text
MeuBotRAG/
├── dados/               # Onde ficam os PDFs para leitura
├── src/                 # Código fonte Python
│   ├── services/        # Lógica de Banco, RAG e WhatsApp
│   └── main.py          # API Principal
├── docker-compose.yml   # Orquestração dos containers
├── Dockerfile           # Configuração da imagem Python
├── requirements.txt     # Dependências do projeto
└── .env                 # (Não comitado) Chaves de API e Senhas