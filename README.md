# 🎓 meuBotRAG — Assistente Académico da UEMA

> Assistente virtual académico integrado no WhatsApp, construído com **Clean Architecture**, **RAG Híbrido** e **memória em três camadas** — tudo a custo **$0**, rodando em hardware comum com Docker.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitectura](#-arquitectura)
- [Pipeline de Processamento](#-pipeline-de-processamento)
- [Stack Tecnológico](#-stack-tecnológico)
- [Estrutura de Pastas](#-estrutura-de-pastas)
- [Pré-requisitos](#-pré-requisitos)
- [Configuração](#-configuração)
- [Como Correr](#-como-correr)
- [Endpoints da API](#-endpoints-da-api)
- [Sistema de Memória](#-sistema-de-memória)
- [RAG Híbrido](#-rag-híbrido)
- [Roteamento Semântico](#-roteamento-semântico)
- [Debug com Chainlit](#-debug-com-chainlit)
- [Roadmap](#-roadmap)

---

## 🔭 Visão Geral

O **meuBotRAG** responde a dúvidas académicas dos alunos da UEMA directamente no WhatsApp. Ele consulta documentos institucionais (editais, calendário académico, guias de contactos) para dar respostas precisas, sem alucinar datas ou siglas.

### O problema que resolve

| Problema | Solução implementada |
|---|---|
| Alucinações em datas e siglas de editais | Busca Híbrida (BM25 + Vetor) com metadados hierárquicos |
| Custos elevados com API de LLM | Gemini 1.5 Flash (Free Tier) + SemanticCache |
| Rate limits no Groq | Migração completa para Google Gemini |
| PostgreSQL/pgvector consome muita RAM | Redis Stack unifica cache, memória e vector store |
| Sem memória entre conversas | Sistema de 3 camadas: Working Memory, Long-Term Facts, Sinais |
| Timeout na Evolution API | Arquitectura assíncrona com fila Celery |

### Comparação antes vs. depois

```
ANTES (v2 — LangChain + Groq + pgvector):
  Mensagem → AgentExecutor → Groq (llama-3.1-8b)
           → tool_calling loop (até 6 iterações)
           → pgvector → resposta
  Custo: ~4.300 tokens/msg | 500–1500ms | risco de rate limit

DEPOIS (v4 — Clean Architecture):
  Mensagem → Guardrails (0 tokens, regex)
           → Working Memory + Long-Term Facts (Redis, 0 tokens, <1ms)
           → Semantic Router (Redis KNN, 0 tokens, ~1ms)
           → Query Transform (Gemini, ~120 tokens, 1 chamada leve)
           → Hybrid Retriever (Redis BM25+Vector, 0 tokens, ~5ms)
           → Gemini Flash (1 chamada limpa, ~950 tokens)
           → Resposta
  Custo: ~1.070 tokens/msg | 800–1200ms | free tier Gemini (1M TPM)
```

---

## 🏛 Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         WhatsApp / Aluno                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ POST /webhook
┌──────────────────────────▼──────────────────────────────────────┐
│                    FastAPI  (bot:9000)                           │
│  DevGuard (dedup + spam) → Celery .delay() → HTTP 200 OK        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Redis Queue (DB 2)
┌──────────────────────────▼──────────────────────────────────────┐
│                   Celery Worker                                  │
│                                                                  │
│  ┌─ handle_message.py ──────────────────────────────────────┐   │
│  │  Guardrails → AgentCore.responder()                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─ AgentCore (9 passos) ────────────────────────────────────┐  │
│  │  1. Working Memory     (Redis, <1ms)                      │  │
│  │  2. Long-Term Facts    (Redis KNN, ~3ms)                  │  │
│  │  3. Semantic Router    (Redis KNN, ~1ms, 0 tokens)        │  │
│  │  4. Query Transform    (Gemini, ~120 tokens)              │  │
│  │  5. Hybrid Retriever   (Redis BM25+Vector, ~5ms)          │  │
│  │  6. Geração Final      (Gemini Flash, ~950 tokens)        │  │
│  │  7. Persistência       (Redis, <1ms)                      │  │
│  │  8. Memory Extractor   (background, não bloqueia)         │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│              Redis Stack  (redis:6379)                           │
│                                                                  │
│  DB 0:  idx:rag:chunks    → Chunks dos PDFs (BM25 + HNSW)       │
│         idx:tools         → Tools para Semantic Router           │
│         mem:work:{id}     → Working Memory (sinais de sessão)    │
│         mem:facts:*       → Long-Term Factual Memory             │
│         chat:{id}         → Histórico de mensagens               │
│         menu_state:{id}   → Estado do menu por utilizador        │
│  DB 2:  Fila Celery                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Pipeline de Processamento

Cada mensagem percorre até 8 passos. Os primeiros 3 não chamam nenhum LLM — poupam tokens e latência.

```
Mensagem do aluno
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ PASSO 0 — Guardrails (regex, 0 tokens, 0ms)                 │
│  Saudações → boas-vindas       | Ofensivos → bloqueio       │
│  Fora de escopo → bloqueio     | Self-RAG signal → precisa_rag │
└────────────────────────────┬────────────────────────────────┘
                             │ bloquear=False
      ┌──────────────────────▼──────────────────────────┐
      │ PASSO 1 — Working Memory (Redis, <1ms)           │
      │  Histórico compactado (sliding window 8 turns)   │
      │  Sinais: última tool, rota, tópico               │
      └──────────────────────┬──────────────────────────┘
                             │
      ┌──────────────────────▼──────────────────────────┐
      │ PASSO 2 — Long-Term Facts (Redis KNN, ~3ms)      │
      │  Busca fatos relevantes do aluno                  │
      │  Ex: "Aluno de Eng. Civil, turno noturno"        │
      └──────────────────────┬──────────────────────────┘
                             │
      ┌──────────────────────▼──────────────────────────┐
      │ PASSO 3 — Semantic Router (Redis KNN, ~1ms)      │
      │  Alta confiança (>0.80) → tool + source_filter   │
      │  Média confiança (0.62–0.80) → tool + doc_type   │
      │  Baixa confiança (<0.62) → Rota.GERAL            │
      └──────────────────────┬──────────────────────────┘
                             │
      ┌──────────────────────▼──────────────────────────┐
      │ PASSO 4 — Query Transform (Gemini, ~120 tokens)  │
      │  Reescreve a pergunta com contexto dos fatos      │
      │  Skip se alta confiança (economiza tokens)        │
      └──────────────────────┬──────────────────────────┘
                             │
      ┌──────────────────────▼──────────────────────────┐
      │ PASSO 5 — Hybrid Retriever (Redis, ~5ms)         │
      │  BM25 (keywords exactas: datas, siglas)          │
      │  Vetor (semântica: intenção da pergunta)          │
      │  RRF (fusão dos ranks)                           │
      │  Fallback step-back se 0 resultados              │
      └──────────────────────┬──────────────────────────┘
                             │
      ┌──────────────────────▼──────────────────────────┐
      │ PASSO 6 — Geração Gemini Flash (~950 tokens)     │
      │  System + Fatos + Histórico + Contexto RAG       │
      │  1 chamada limpa, sem tool-calling loop          │
      └──────────────────────┬──────────────────────────┘
                             │
      ┌──────────────────────▼──────────────────────────┐
      │ PASSO 7 — Persistência (Redis, <1ms)             │
      │  Salva turn no histórico da sessão               │
      └──────────────────────┬──────────────────────────┘
                             │
      ┌──────────────────────▼──────────────────────────┐
      │ PASSO 8 — Memory Extractor (background)          │
      │  Analisa o turn e extrai novos fatos             │
      │  Não bloqueia a resposta ao utilizador           │
      └─────────────────────────────────────────────────┘
```

---

## 🛠 Stack Tecnológico

| Componente | Tecnologia | Função |
|---|---|---|
| **LLM** | Google Gemini 2.0 Flash (Free Tier) | Geração de respostas, query transform, extracção de fatos |
| **Embeddings** | BAAI/bge-m3 (local, CPU) | Vectores de 1024 dims para busca semântica |
| **Vector Store** | Redis Stack (HNSW) | Chunks dos PDFs + tool routing |
| **BM25** | Redis Stack (RediSearch) | Busca por keywords exactas |
| **Fila** | Celery + Redis | Processamento assíncrono sem timeout |
| **API** | FastAPI + Uvicorn | Webhook WhatsApp |
| **WhatsApp** | Evolution API | Gateway de mensagens |
| **Parser PDF** | LlamaParse (cloud) / PyMuPDF (local) | Extracção de texto dos editais |
| **Debug** | Chainlit | Interface de teste visual |
| **Infra** | Docker + Docker Compose | Orquestração de containers |

### Recursos de hardware necessários

| Recurso | Mínimo | Recomendado |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 6+ cores |
| GPU | Não necessária (CPU-only) | Qualquer (sem CUDA obrigatório) |
| Disco | 5 GB | 10 GB |

> ✅ Testado e desenvolvido num PC com **AMD RX 580** e **16 GB RAM** — sem CUDA, sem GPU dedicada para IA.

---

## 📁 Estrutura de Pastas

```
meuBotRAG/
│
├── src/
│   ├── agent/
│   │   ├── core.py               ← Orquestrador principal (9 passos)
│   │   └── prompts.py            ← SYSTEM_UEMA, prompts, few-shot examples
│   │
│   ├── application/
│   │   ├── handle_message.py     ← Ponte WhatsApp ↔ AgentCore
│   │   └── tasks.py              ← Tasks Celery (processar_mensagem_task)
│   │
│   ├── domain/
│   │   ├── entities.py           ← AgentResponse, Rota, Mensagem (dataclasses)
│   │   ├── guardrails.py         ← Guardrails: greeter, block list, self-RAG signal
│   │   ├── menu.py               ← Navegação de menu estático (regex, stateless)
│   │   └── semantic_router.py    ← Roteamento por similaridade vetorial (Redis KNN)
│   │
│   ├── infrastructure/
│   │   ├── celery_app.py         ← Configuração Celery
│   │   ├── observability.py      ← Logs estruturados e métricas
│   │   ├── redis_client.py       ← Cliente Redis singleton + índices BM25/HNSW
│   │   └── settings.py           ← Variáveis de ambiente (Pydantic Settings)
│   │
│   ├── memory/
│   │   ├── long_term_memory.py   ← Fatos do utilizador (Redis JSON + KNN)
│   │   ├── memory_extractor.py   ← Extracção de fatos em background (Gemini)
│   │   ├── redis_memory.py       ← Histórico legacy + estado do menu
│   │   └── working_memory.py     ← Sessão activa (sinais, histórico compactado)
│   │
│   ├── middleware/
│   │   └── dev_guard.py          ← Dedup, spam guard, whitelist de dev
│   │
│   ├── providers/
│   │   └── gemini_provider.py    ← Cliente Gemini + retry + structured outputs
│   │
│   ├── rag/
│   │   ├── embeddings.py         ← Modelo BAAI/bge-m3 singleton (CPU)
│   │   ├── hybrid_retriever.py   ← BM25 + Vector → RRF → contexto hierárquico
│   │   ├── ingestion.py          ← PDF → chunks → Redis (com manifesto de hash)
│   │   └── query_transform.py    ← Step-back, rewrite, sub-queries (Gemini)
│   │
│   ├── services/
│   │   └── evolution_service.py  ← Envio de mensagens via Evolution API
│   │
│   └── tools/
│       ├── __init__.py           ← get_tools_ativas()
│       ├── calendar_tool.py      ← Tool: calendário académico
│       ├── tool_contatos.py      ← Tool: contactos UEMA
│       └── tool_edital.py        ← Tool: edital PAES 2026
│
├── debug/
│   └── debug_chainlit.py         ← Painel de debug visual (Chainlit)
│
├── dados/                        ← PDFs institucionais (não comitar)
│   ├── calendario-academico-2026.pdf
│   ├── edital_paes_2026.pdf
│   ├── guia_contatos_2025.pdf
│   ├── contatos_saoluis.txt
│   └── regras_ru.txt
│
├── .env.example                  ← Template de configuração
├── docker-compose.yml            ← 4 serviços: redis, evolution-postgres, bot, celery-worker
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## 📦 Pré-requisitos

- **Docker** e **Docker Compose** v2+
- **Python 3.11** ou 3.12 (para o painel de debug local)
- Conta Google AI Studio com API Key do Gemini (gratuita)
- Conta LlamaIndex Cloud (opcional, para melhor extracção de PDFs complexos)
- **Evolution API** configurada e acessível

---

## 🔧 Configuração

### 1. Clonar e preparar o projecto

```bash
git clone https://github.com/seu-usuario/meuBotRAG.git
cd meuBotRAG
cp .env.example .env
```

### 2. Configurar o `.env`

```env
# ── LLM (obrigatório) ────────────────────────────────────────────────────────
GEMINI_API_KEY=sua_chave_aqui          # https://aistudio.google.com
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMP=0.3
GEMINI_MAX_TOKENS=1024

# ── HuggingFace (opcional, evita rate limit no download do modelo) ───────────
HF_TOKEN=hf_xxxxxxxxxxxx              # https://huggingface.co/settings/tokens

# ── Parser de PDF ────────────────────────────────────────────────────────────
# "pymupdf"    → gratuito, local, rápido (recomendado para começar)
# "llamaparse" → pago ($0.003/pág), melhor para tabelas complexas
PDF_PARSER=pymupdf
LLAMA_CLOUD_API_KEY=                   # só necessário se PDF_PARSER=llamaparse

# ── Redis Stack ──────────────────────────────────────────────────────────────
REDIS_URL=redis://redis:6379/0         # dentro do Docker
# REDIS_URL=redis://localhost:6379/0   # para debug local

# ── Evolution API (WhatsApp) ─────────────────────────────────────────────────
EVOLUTION_BASE_URL=http://evolution-api:8080
EVOLUTION_API_KEY=sua_chave_evolution
EVOLUTION_INSTANCE_NAME=meubot
WHATSAPP_HOOK_URL=http://bot:9000/webhook

# ── Dev / Debug ──────────────────────────────────────────────────────────────
DEV_MODE=false
DEV_WHITELIST=5598999999999            # números permitidos em modo dev (separados por vírgula)
LOG_LEVEL=INFO
```

### 3. Adicionar os PDFs

Coloca os documentos institucionais na pasta `dados/`:

```
dados/
├── calendario-academico-2026.pdf   ← obrigatório
├── edital_paes_2026.pdf            ← obrigatório
├── guia_contatos_2025.pdf          ← obrigatório
├── contatos_saoluis.txt            ← opcional
└── regras_ru.txt                   ← opcional
```

> ⚠️ Os PDFs são ingeridos automaticamente no primeiro arranque. O processo demora 2–10 minutos dependendo do tamanho e do parser escolhido.

---

## 🚀 Como Correr

### Produção (Docker Compose)

```bash
# Iniciar todos os serviços
docker compose up -d

# Ver logs em tempo real
docker compose logs -f bot celery-worker

# Verificar saúde
curl http://localhost:9000/health
```

### Verificar se a ingestão correu bem

```bash
curl http://localhost:9000/banco/sources
# Deve retornar a lista de PDFs ingeridos com número de chunks
```

### Parar os serviços

```bash
docker compose down
# Para apagar também os dados do Redis:
docker compose down -v
```

### Desenvolvimento com hot-reload

O `docker-compose.yml` já inclui volumes `./src:/app/src` e `./dados:/app/dados` — qualquer mudança no código reflecte imediatamente sem rebuild.

---

## 🌐 Endpoints da API

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/webhook` | Recebe eventos da Evolution API |
| `GET` | `/health` | Status do sistema (Redis, AgentCore, modelo) |
| `GET` | `/logs?limit=20` | Últimos erros registados |
| `GET` | `/metrics?limit=50` | Métricas de uso (tokens, latência, rotas) |
| `GET` | `/banco/sources` | PDFs ingeridos no Redis |
| `GET` | `/fatos/{user_id}` | Fatos long-term de um utilizador |
| `GET` | `/memoria/{session_id}` | Working memory de uma sessão |
| `DELETE` | `/memoria/{session_id}` | Limpa a sessão de um utilizador |

---

## 🧠 Sistema de Memória

O bot implementa **3 camadas de memória**, inspiradas na arquitectura do [Agent Memory Server (Redis)](https://github.com/redis/agent-memory-server):

### Camada 1 — Working Memory (sessão activa)

```
Redis Key: chat:{session_id}
TTL: 30 minutos de inactividade
```

Guarda as últimas mensagens da conversa activa com **sliding window de 8 turns**. Garante que o Gemini recebe apenas o histórico relevante, sem estourar o context window.

### Camada 2 — Sinais de Contexto

```
Redis Key: mem:work:{session_id}
TTL: 30 minutos
```

Hash com metadados rápidos da sessão: `ultima_tool`, `rota`, `ultimo_topico`, `confianca_routing`. Usados para optimizar o roteamento nas mensagens seguintes.

### Camada 3 — Long-Term Factual Memory

```
Redis Key (lista):  mem:facts:list:{user_id}
Redis Key (vector): mem:facts:vec:{user_id}:{hash}
TTL: 30 dias
```

Fatos extraídos de conversas passadas pelo `memory_extractor.py`. Exemplos reais:
- `"Aluno do curso de Engenharia Civil, turno noturno"`
- `"Inscrito no PAES 2026 na categoria BR-PPI"`
- `"Dúvida recorrente sobre trancamento de matrícula"`

Estes fatos são injectados no prompt de geração, permitindo respostas personalizadas sem o aluno ter de se repetir.

**Extracção em background:** O `memory_extractor.py` corre após cada resposta sem bloquear o fluxo principal. Usa Gemini com Structured Output (`ExtracaoFatosSchema`) para garantir que apenas fatos verificáveis são guardados, com cooldown de 2 minutos entre extrações.

---

## 🔍 RAG Híbrido

### Por que a busca simples por vector alucina datas e siglas

Embeddings capturam **semântica** mas não **exactidão lexical**. A query `"matrícula veteranos"` pode recuperar um chunk de 2025 porque é semanticamente similar — o modelo não distingue anos.

### Como a busca híbrida resolve isto

```
Query: "matrícula veteranos UEMA 2026.1 data período"
         │                           │
         ▼                           ▼
    Embedding                    BM25 (keyword)
    (semântica)                  (exactidão)
         │                           │
         └──────────── RRF ──────────┘
                        │
                        ▼
          EVENTO: Matrícula de veteranos
          DATA: 03/02/2026 a 07/02/2026
          SEM: 2026.1
          [FONTE: Calendário Académico UEMA 2026]
```

O **BM25** garante que `"2026.1"`, `"veteranos"`, `"BR-PPI"` e outras siglas são encontradas por match exacto, enquanto o **vetor** garante que a intenção semântica é compreendida. O **RRF (Reciprocal Rank Fusion)** funde os dois rankings para o melhor resultado.

### Chunking hierárquico anti-alucinação

Cada chunk é formatado com um cabeçalho que ancora o LLM:

```
[EDITAL PAES 2026 | edital]
CURSO: Engenharia Civil | TURNO: Noturno | AC: 40 | PcD: 2 | TOTAL: 42
```

O Gemini vê explicitamente de onde veio a informação antes do conteúdo — estudos de RAG mostram que cabeçalhos de fonte reduzem alucinações em ~40%.

---

## 🗺 Roteamento Semântico

O `semantic_router.py` decide qual documento consultar **sem chamar o LLM** — apenas por similaridade vectorial no Redis.

```
Mensagem: "quando começa a matrícula?"
    │
    ▼
Embedding da mensagem (CPU, ~10ms)
    │
    ▼
Redis KNN: encontra tool mais similar
    │
    ├── score > 0.80 → ALTA CONFIANÇA
    │   → Rota.CALENDARIO + source_filter="calendario-academico-2026.pdf"
    │   → Skip do Query Transform (economiza ~120 tokens)
    │
    ├── score 0.62–0.80 → MÉDIA CONFIANÇA
    │   → Rota.CALENDARIO + doc_type="calendario"
    │   → Executa Query Transform para enriquecer a query
    │
    └── score < 0.62 → BAIXA CONFIANÇA
        → Rota.GERAL → Gemini decide livremente
```

**Tools registadas:**
- `consultar_calendario_academico` → Rota.CALENDARIO
- `consultar_edital_paes_2026` → Rota.EDITAL
- `consultar_contatos_uema` → Rota.CONTATOS

---

## 🐛 Debug com Chainlit

O painel de debug corre localmente e conecta ao mesmo Redis que o Docker.

### Configuração

```bash
# 1. Instalar dependências de debug
pip install chainlit tiktoken google-genai redis[hiredis]

# 2. Criar .env.local para apontar ao Redis do Docker
cat > .env.local << EOF
REDIS_URL=redis://localhost:6379/0
GEMINI_API_KEY=sua_chave_aqui
EOF

# 3. Correr o Chainlit (sempre da raiz do projecto)
chainlit run debug/debug_chainlit.py --port 8001
```

### Comandos disponíveis no painel

| Comando | Descrição |
|---|---|
| `/status` | Mostra estado do Redis, AgentCore e PDFs ingeridos |
| `/fatos` | Lista todos os fatos long-term do utilizador de debug |
| `/extracao` | Força extracção de fatos da conversa actual |
| `/router <query>` | Testa o semantic router para uma query |
| `/limpar` | Limpa sessão e histórico |
| `/ingerir` | Força re-ingestão dos PDFs |
| `/exportar` | Exporta log da sessão |
| `/ajuda` | Lista todos os comandos |

> ⚠️ Requer Python 3.11 ou 3.12 — o Chainlit não suporta Python 3.13+.

---

## 🗓 Roadmap

### Próximas melhorias planeadas

- [ ] **Semantic Cache** — guardar respostas a perguntas frequentes no Redis (economia estimada: ~34% dos tokens)
- [ ] **Guardrails activos** — activar `guardrails.py` no `handle_message.py` e remover `menu.py` / `router.py`
- [ ] **Self-RAG** — skip do retriever para perguntas que não precisam de documentos
- [ ] **Corrective RAG (CRAG)** — re-busca automática se o score RRF dos chunks for < 0.35
- [ ] **Few-shot examples no system prompt** — 3–5 exemplos Q&A para reduzir alucinações de formato
- [ ] **Modelo local** — Qwen2.5-7B-Instruct quantizado Q4_K_M (~4.5GB RAM) via llama.cpp para $0 absoluto

### Melhorias de arquitectura futuras

- [ ] **Gateway Go** — separar o webhook HTTP (Go, alta concorrência) do worker de IA (Python) para cenários multi-tenant ou alto volume
- [ ] **Multi-tenant** — suporte a múltiplas instituições (UFMA, UFRJ, etc.) com o mesmo código

---

## 📄 Licença

Este projecto é académico e de uso interno. Consulte o ficheiro `LICENSE` para os termos completos.

---

<div align="center">

Construído com ❤️ para os alunos da UEMA

`FastAPI` · `Gemini Flash` · `Redis Stack` · `BAAI/bge-m3` · `Celery` · `Evolution API` · `Docker`

</div>