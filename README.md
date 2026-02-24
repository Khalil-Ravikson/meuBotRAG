# 🎓 Bot UEMA — Assistente Virtual WhatsApp com RAG

Assistente virtual do WhatsApp da **UEMA Campus Paulo VI, São Luís-MA**, com arquitetura **Multi-step Agentic RAG** e **Clean Architecture** em 6 camadas.

Responde perguntas sobre o Calendário Acadêmico 2026, Edital PAES 2026 e Contatos Institucionais usando PDFs ingeridos em um banco vetorial (pgvector), com LLM via Groq e histórico de conversas no Redis.

---

## Índice

1. [Pré-requisitos](#1-pré-requisitos)
2. [O `.env` — entendendo de uma vez por todas](#2-o-env--entendendo-de-uma-vez-por-todas)
3. [config.py vs settings.py — qual usar?](#3-configpy-vs-settingspy--qual-usar)
4. [Quickstart — subindo em 5 passos](#4-quickstart--subindo-em-5-passos)
5. [Desenvolvimento local sem Docker](#5-desenvolvimento-local-sem-docker)
6. [Arquitetura — as 6 camadas](#6-arquitetura--as-6-camadas)
7. [Estrutura de pastas](#7-estrutura-de-pastas)
8. [Descrição de cada arquivo](#8-descrição-de-cada-arquivo)
9. [Como uma mensagem é processada — pipeline completo](#9-como-uma-mensagem-é-processada--pipeline-completo)
10. [Pipeline de testes](#10-pipeline-de-testes)
11. [Painel de debug — Chainlit](#11-painel-de-debug--chainlit)
12. [LangSmith — rastreamento do agente](#12-langsmith--rastreamento-do-agente)
13. [Perguntas frequentes](#13-perguntas-frequentes)

---

## 1. Pré-requisitos

| Ferramenta | Versão | Para que serve |
|---|---|---|
| Docker + Docker Compose | 24+ | Rodar todos os serviços |
| Python | 3.11 ou 3.12 | Dev local, testes, Chainlit |
| ngrok ou domínio público | — | Expor o bot ao WhatsApp |

**Contas necessárias:**

| Serviço | Link | Plano |
|---|---|---|
| Groq (LLM) | [console.groq.com](https://console.groq.com) | Free |
| LlamaCloud (parse PDFs) | [cloud.llamaindex.ai](https://cloud.llamaindex.ai) | Free |
| HuggingFace (embedding) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free |
| LangSmith (observabilidade) | [smith.langchain.com](https://smith.langchain.com) | Free até 5k traces/mês |

---

## 2. O `.env` — entendendo de uma vez por todas

O `.env` é um arquivo de texto simples com uma variável por linha:

```
GROQ_API_KEY=gsk_...
DB_USER=postgres
REDIS_URL=redis://localhost:6379/0
```

Ele guarda suas chaves de API **fora do código** — o git ignora este arquivo via `.gitignore`. Você versiona o `.env.example` (com valores fictícios) e cria o `.env` real só na sua máquina.

### Como o `.env` chega a cada componente

```
.env  (arquivo no seu computador)
  │
  ├─── Docker Compose lê automaticamente ──────────────────────────────────┐
  │                                                                         │
  │  Uso 1: Interpolação no docker-compose.yml                              │
  │  Antes de subir, o Compose substitui ${DB_USER} pelo valor do .env     │
  │  Ex: DATABASE_URL=postgresql+psycopg://${DB_USER}:${DB_PASS}@db:5432/  │
  │       ↓ vira ↓                                                          │
  │      DATABASE_URL=postgresql+psycopg://postgres:senha@db:5432/          │
  │                                                                         │
  │  Uso 2: env_file: .env no serviço bot                                   │
  │  Injeta o .env completo DENTRO do container em runtime                  │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
                          dentro do container
                                    │
  ┌─────────────────────────────────▼──────────────────────────────────────┐
  │  src/infrastructure/settings.py (pydantic-settings)                    │
  │  Lê variáveis de ambiente → settings.GROQ_API_KEY, settings.REDIS_URL  │
  └─────────────────────────────────────────────────────────────────────────┘
```

### Por que algumas variáveis aparecem tanto no `.env` quanto no `environment:` do docker-compose?

Três variáveis usam **nomes de serviço Docker** como host (`db`, `redis`, `waha`) — não `localhost`. No seu `.env` você tem `localhost` para funcionar em dev local. O `docker-compose.yml` sobrescreve essas três especificamente:

```yaml
environment:
  - DATABASE_URL=postgresql+psycopg://${DB_USER}:${DB_PASS}@db:5432/${DB_NAME}
  - REDIS_URL=redis://redis:6379/0
  - WAHA_BASE_URL=http://waha:3000
```

Tudo o mais (GROQ_API_KEY, HF_TOKEN, LLAMA_CLOUD_API_KEY etc.) vem direto do `env_file: .env`.

### Criando o `.env`

```bash
cp .env.example .env
nano .env   # preencha com seus valores reais
```

---

## 3. config.py vs settings.py — qual usar?

**Use `settings.py`. Delete o `config.py`.**

| Característica | `config.py` (antigo) | `settings.py` (novo) |
|---|---|---|
| Lê o `.env` | `os.getenv()` sem validação | Pydantic valida tipo automaticamente |
| Valor inválido | Silencioso (bug tarde) | Falha no startup com mensagem clara |
| `print()` de debug | Sim — vaza em produção | Não |
| Testável | Difícil | `Settings(_env_file="tests/.env.test")` |
| Singleton | Não | `@lru_cache` — instanciado uma vez |

**Como migrar em 30 segundos:**

```bash
# Troque em todos os arquivos que ainda usam config.py:
grep -r "from src.config import" src/
```

```python
# Antes (apague o config.py depois da troca)
from src.config import settings

# Depois
from src.infrastructure.settings import settings
```

Os nomes das variáveis são idênticos — `settings.GROQ_API_KEY`, `settings.REDIS_URL` etc.

---

## 4. Quickstart — subindo em 5 passos

```bash
# 1. Clone e configure
git clone <repo-url> meuBotRAG
cd meuBotRAG
cp .env.example .env
nano .env          # preencha GROQ_API_KEY, DB_PASS, WAHA_API_KEY, LLAMA_CLOUD_API_KEY

# 2. Coloque os PDFs na pasta dados/
# Os nomes devem ser EXATAMENTE esses (case sensitive):
ls dados/
# calendario-academico-2026.pdf
# edital_paes_2026.pdf
# guia_contatos_2025.pdf

# 3. Suba todos os serviços
docker-compose up -d --build

# 4. Acompanhe o startup (aguarde ~2 minutos — ingestão dos PDFs)
docker-compose logs -f bot

# 5. Verifique se está tudo ok
curl http://localhost:8000/health
# {"status":"ok","redis":true,"agente":true,"dev_mode":false}

curl http://localhost:8000/banco/sources
# Deve mostrar os 3 PDFs ingeridos
```

**Para expor ao WhatsApp via ngrok:**

```bash
ngrok http 8000
# Copie a URL HTTPS gerada, ex: https://abc123.ngrok.io
# No .env, atualize: WHATSAPP_HOOK_URL=https://abc123.ngrok.io/webhook
# Reinicie o bot: docker-compose restart bot
```

---

## 5. Desenvolvimento local sem Docker

```bash
# 1. Python 3.11
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Dependências
pip install -r requirements.txt

# 3. Sobe só a infra (banco + redis) via Docker
docker-compose up -d db redis

# 4. Ajusta o .env para localhost
# DATABASE_URL=postgresql+psycopg://postgres:senha@localhost:5433/vectordb
# REDIS_URL=redis://localhost:6379/0

# 5. Roda o bot
uvicorn src.main:app --reload --port 8000

# 6. Painel de debug (outra janela de terminal)
pip install chainlit tiktoken
chainlit run debug/debug_chainlit.py --port 8001
```

---

## 6. Arquitetura — as 6 camadas

O projeto segue **Clean Architecture**. A regra fundamental: **camadas internas nunca importam camadas externas**.

```
┌─────────────────────────────────────────────────────────┐
│  INTERFACE        src/api/                               │
│  FastAPI routes, schemas Pydantic, endpoints HTTP        │
├─────────────────────────────────────────────────────────┤
│  APPLICATION      src/application/                       │
│  Casos de uso: orquestra as camadas abaixo               │
│  handle_webhook → handle_message                         │
├─────────────────────────────────────────────────────────┤
│  AGENT            src/agent/                             │
│  AgentExecutor LangChain, state, prompts, validação      │
├─────────────────────────────────────────────────────────┤
│  DOMAIN           src/domain/           ← SEM I/O        │
│  Entidades, menu stateless, router por regex             │
│  Testável com assert puro, sem nenhum mock               │
├─────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE   src/infrastructure/  src/memory/       │
│                   src/rag/             src/providers/    │
│  Redis, pgvector, LLM, settings, observabilidade         │
├─────────────────────────────────────────────────────────┤
│  EXTERNAL         WAHA · Groq · pgvector · Redis         │
│  Serviços externos — nunca importados pela camada Domain │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Estrutura de pastas

```
meuBotRAG/
│
├── dados/                              # PDFs para ingestão (não vai ao git)
│   ├── calendario-academico-2026.pdf
│   ├── edital_paes_2026.pdf
│   └── guia_contatos_2025.pdf
│
├── debug/                              # Ferramentas de desenvolvimento
│   ├── debug_chainlit.py               # Painel interativo (sem WhatsApp)
│   └── chainlit.toml                   # Visual do painel (não vai ao Docker)
│
├── src/                                # Código-fonte da aplicação
│   ├── main.py                         # Bootstrap FastAPI
│   │
│   ├── api/                            # Camada de interface
│   │   └── schemas.py                  # Modelos Pydantic de request/response
│   │
│   ├── application/                    # Casos de uso
│   │   ├── handle_webhook.py           # Recebe e valida payload WAHA
│   │   └── handle_message.py           # Decide: menu direto ou agente
│   │
│   ├── agent/                          # Núcleo do agente LangChain
│   │   ├── core.py                     # AgentExecutor + histórico Redis
│   │   ├── state.py                    # AgentState: objeto de trabalho
│   │   ├── prompts.py                  # Todos os prompts (fonte única)
│   │   └── validator.py                # Valida output antes de enviar
│   │
│   ├── domain/                         # Regras de negócio puras — SEM I/O
│   │   ├── entities.py                 # Mensagem, AgentResponse, Rota, EstadoMenu
│   │   ├── menu.py                     # Lógica de menu (stateless, testável)
│   │   └── router.py                   # Roteamento por intenção (regex puro)
│   │
│   ├── rag/                            # Retrieval-Augmented Generation
│   │   ├── vector_store.py             # Singleton pgvector + embedding BAAI/bge-m3
│   │   └── ingestor.py                 # LlamaParse + chunking + salva no banco
│   │
│   ├── tools/                          # Tools do agente LangChain
│   │   ├── __init__.py                 # Lista de tools ativas
│   │   ├── tool_calendario.py          # Busca datas no pgvector
│   │   ├── tool_edital.py              # Busca regras do PAES no pgvector
│   │   └── tool_contatos.py            # Busca contatos no pgvector
│   │
│   ├── services/                       # Integrações externas
│   │   └── waha_service.py             # HTTP client do WAHA
│   │
│   ├── providers/                      # Provedores de LLM
│   │   └── groq_provider.py            # ChatGroq com retry no 429
│   │
│   ├── infrastructure/                 # Configuração e clientes de infra
│   │   ├── settings.py                 # Pydantic Settings — lê o .env
│   │   ├── redis_client.py             # Singleton Redis compartilhado
│   │   └── observability.py            # Logs estruturados + métricas
│   │
│   ├── memory/                         # Histórico de conversas
│   │   └── redis_memory.py             # LangChain history + estado menu
│   │
│   └── middleware/                     # Filtros de segurança
│       └── dev_guard.py                # Whitelist, dedup, validação WAHA
│
├── tests/
│   ├── unit/                           # Sem Docker, sem mocks de infra
│   │   ├── test_menu.py
│   │   ├── test_router.py
│   │   └── test_validator.py
│   ├── integration/                    # Com Redis e pgvector reais
│   └── e2e/                            # Fluxo completo com Groq mockado
│
├── docker-compose.yml                  # Orquestra: waha + db + redis + bot
├── Dockerfile                          # Imagem do bot
├── requirements.txt                    # Dependências Python
├── pyproject.toml                      # Config de pytest e metadados
├── .env.example                        # Template do .env (vai ao git)
├── .env                                # Suas chaves reais (NÃO vai ao git)
└── .gitignore
```

---

## 8. Descrição de cada arquivo

### `src/main.py`

Ponto de entrada da aplicação FastAPI. Configura logging, filtra ruído de logs do uvicorn e httpx, e no evento de startup executa em sequência: (1) ingestão dos PDFs se o banco estiver vazio, (2) diagnóstico dos sources se `DEV_MODE=true`, (3) inicialização do agente com as tools ativas, (4) inicialização do WAHA com configuração do webhook.

Endpoints: `POST /webhook`, `GET /health`, `GET /logs`, `GET /metrics`, `GET /banco/sources`.

---

### `src/infrastructure/settings.py`

**Substitui completamente o `config.py`.** Usa `pydantic-settings` para ler o `.env` com validação automática de tipos. É um singleton via `@lru_cache` — instanciado uma única vez em todo o processo. Se uma variável obrigatória estiver ausente ou com tipo errado, o processo falha no startup com mensagem clara.

Importe em qualquer módulo:
```python
from src.infrastructure.settings import settings
print(settings.GROQ_API_KEY)
```

### `src/infrastructure/redis_client.py`

Singleton do cliente Redis. Um único pool de conexões compartilhado por `memory/`, `middleware/` e `infrastructure/observability.py`. Sem instâncias espalhadas. `get_redis()` retorna sempre o mesmo objeto. `redis_ok()` retorna `bool` sem lançar exceção — usado no `/health`.

### `src/infrastructure/observability.py`

Substitui o `logger_service.py`. Singleton `obs` com métodos `obs.error()`, `obs.warn()`, `obs.info()` que logam no terminal **e** salvam no Redis com `ltrim` (mantém últimos 100 por nível). `obs.registrar_resposta()` salva métricas de tokens, latência e iterações para análise via `/metrics`.

---

### `src/domain/entities.py`

Tipos puros de domínio — sem Redis, sem Groq, sem nenhum import externo. São os tipos que trafegam entre todas as camadas. Contém: `Rota` (enum: CALENDARIO, EDITAL, CONTATOS, GERAL), `EstadoMenu` (enum: MAIN, SUB_CALENDARIO, SUB_EDITAL, SUB_CONTATOS), `Mensagem` (dados brutos do WhatsApp), `RAGResult` (chunk retornado pelo pgvector), `AgentResponse` (resposta final do agente).

### `src/domain/menu.py`

Lógica de menu **100% stateless**. Recebe `(texto, estado_atual)` e retorna um `dict` indicando se a resposta é um menu direto ou deve ir para o LLM. Sem Redis, sem I/O. O estado vem injetado por `handle_message.py`. Testável com um simples `assert` sem nenhum mock. Contém os textos dos menus, opções numéricas expandidas em perguntas completas para o LLM, e regex de saudações/voltar.

### `src/domain/router.py`

Roteamento por intenção usando regex puro. Recebe `(texto, estado)` e retorna uma `Rota`. Sem I/O. O padrão EDITAL é avaliado **antes** do CALENDARIO para resolver a ambiguidade de "data de inscrição do PAES" (→ EDITAL, não CALENDARIO). Se o usuário está em um submenu ativo, a rota é forçada por esse submenu independente do texto.

---

### `src/agent/core.py`

Orquestra o agente LangChain. No método `inicializar(tools)`: monta `ChatGroq`, `ChatPromptTemplate` com o `SYSTEM_PROMPT`, `create_tool_calling_agent`, `AgentExecutor` (com `max_iterations` e `max_execution_time` do settings), e `RunnableWithMessageHistory` ligado ao `get_historico_limitado` do Redis. Ativa o LangSmith se `settings.langsmith_ativo`. No método `responder(state)`: invoca o agente, valida o output, registra métricas. Trata dois erros críticos: 429 (rate limit Groq) com mensagem amigável, e `tool_use_failed` (histórico corrompido) limpando o Redis e retentando sem histórico.

### `src/agent/state.py`

`AgentState` é o objeto de trabalho que carrega todo o contexto de uma execução: identificação do usuário, rota detectada, prompt enriquecido, contador de iterações, tokens de entrada/saída, resultados RAG acumulados e timestamp de início para calcular latência. Dataclass Python puro — sem I/O.

### `src/agent/prompts.py`

**Fonte única de todos os prompts.** Nenhum outro arquivo deve ter strings de system prompt. Contém: `SYSTEM_PROMPT` (prompt principal do agente com todas as regras e descrição das tools), `_CONTEXTOS` (dict de Rota → instrução específica para aquela área), `montar_prompt_enriquecido()` (combina rota + contexto do usuário + mensagem), mensagens de erro amigáveis (`MSG_RATE_LIMIT`, `MSG_ERRO_TECNICO`, `MSG_NAO_ENCONTRADO`), e `OUTPUTS_INVALIDOS` (frozenset de strings internas do LangChain que jamais devem ir ao usuário).

### `src/agent/validator.py`

Última barreira antes de enviar ao WhatsApp. Verifica: output não é uma string interna do LangChain ("Agent stopped due to max iterations." etc.), output tem mais de 10 caracteres, output não é vazio ou só espaço. Retorna `ValidationResult(valido, output, motivo)`. Puro, sem I/O, testável com `assert`.

---

### `src/rag/vector_store.py`

Singleton do modelo de embedding (BAAI/bge-m3, ~1.3GB) e da conexão com pgvector. O modelo é carregado **uma única vez** via `@lru_cache` — chamadas de múltiplas tools reutilizam a mesma instância sem custo. Configura `HF_TOKEN` no ambiente antes do download para evitar rate limit do HuggingFace Hub. O parâmetro `normalize_embeddings=True` melhora a similaridade coseno.

### `src/rag/ingestor.py`

Processa os arquivos da pasta `dados/`. O dict `PDF_CONFIG` é a fonte única de verdade: mapeia nome exato do arquivo para instrução de parsing e parâmetros de chunking. PDFs são parseados com LlamaParse usando instrução específica por arquivo (tabelas de calendário, vagas do edital, contatos). TXTs são lidos diretamente sem LlamaParse. O metadado `source` salvo no banco é o nome exato do arquivo — deve bater com `SOURCE_*` em cada tool.

---

### `src/tools/__init__.py`

Registra as tools ativas via `get_tools_ativas()`. Para adicionar uma nova tool: crie o arquivo em `src/tools/`, importe a fábrica aqui, adicione à lista.

### `src/tools/tool_calendario.py`

Busca eventos do Calendário Acadêmico 2026 no pgvector filtrado por `source = "calendario-academico-2026.pdf"`. Usa retriever MMR com `k=4`, `fetch_k=25`, `lambda_mult=0.75` (75% relevância, 25% diversidade). Normaliza a query removendo acentos antes da busca.

### `src/tools/tool_edital.py`

Busca regras e vagas do Edital PAES 2026 filtrado por `source = "edital_paes_2026.pdf"`. Usa retriever `similarity` (não MMR) porque as seções do edital são bem distintas — queremos os chunks mais similares à query, não diversidade.

### `src/tools/tool_contatos.py`

Busca contatos institucionais filtrado por `source = "guia_contatos_2025.pdf"`. Usa MMR com `lambda_mult=0.65` (mais diversidade que o calendário) para trazer contatos de **setores diferentes** quando a query é ampla ("contatos do CECEN" deve retornar vários coordenadores, não o mesmo repetido).

---

### `src/application/handle_webhook.py`

Ponto de entrada de toda mensagem recebida. Recebe o payload bruto do WAHA, chama `DevGuard.validar()`, converte o resultado para a entidade `Mensagem` e chama `handle_message()`. Retorna `{"status": "ok"}` sempre (WAHA não precisa de resposta específica).

### `src/application/handle_message.py`

Orquestrador principal. Fluxo: (1) carrega estado do menu do Redis, (2) chama `domain/menu.processar_mensagem()` (stateless), (3) se resposta de menu direto → envia sem LLM, (4) se ação → chama `domain/router.analisar()`, monta prompt enriquecido, cria `AgentState` e chama `agent_core.responder()`, (5) persiste contexto, (6) envia resposta via WAHA.

---

### `src/memory/redis_memory.py`

Gerencia três tipos de dados no Redis:

**Histórico de conversa** (TTL 30min): usa `RedisChatMessageHistory` do LangChain com duas camadas de proteção — sanitização de `tool_calls` órfãos (quando o Groq retorna 400/`tool_use_failed`, a AIMessage com tool_calls fica no Redis sem o ToolMessage correspondente, corrompendo as próximas chamadas) e sliding window de 20 mensagens com corte sempre em `HumanMessage` (nunca no meio de um par tool).

**Estado do menu** (TTL 30min): qual submenu o usuário está navegando. Persiste entre mensagens para que "1" no `SUB_EDITAL` signifique "vagas AC" e não a opção 1 do menu principal.

**Contexto do usuário** (TTL 1h): última intenção, nome, curso — para enriquecer o prompt.

### `src/services/waha_service.py`

HTTP client async para o WAHA usando `httpx`. Métodos: `enviar_mensagem(chat_id, texto)`, `verificar_sessao()`, `configurar_webhook()`, `inicializar()` (chamado no startup). Todos com tratamento de `ConnectError` e `TimeoutException`.

### `src/middleware/dev_guard.py`

"Porteiro" de toda mensagem. Valida em ordem: evento é `"message"`, não é `fromMe`, `chat_id` existe e é válido, não é grupo (`@g.us`), não é status broadcast. Em `DEV_MODE=true`: sender_phone deve estar na `DEV_WHITELIST`. Deduplica via Redis (TTL 5min): mesmo `event_id` não é processado duas vezes.

### `src/providers/groq_provider.py`

Singleton do `ChatGroq` com retry automático em erro 429 (rate limit) usando backoff exponencial: espera 2s, depois 4s, depois 8s entre tentativas. Evita que uma rajada de mensagens simultâneas quebre o agente.

---

### `debug/debug_chainlit.py`

Painel interativo para testar o agente sem precisar do WhatsApp. Usa os mesmos módulos de produção (`agent_core`, `domain/menu`, `domain/router`, `redis_memory`). Exibe a rota detectada e o estado do menu como metadados. Comandos disponíveis no chat: `/ajuda`, `/status`, `/limpar`, `/diagnostico`, `/modo agente`, `/modo direto`, `/ingerir`, `/exportar`.

### `debug/chainlit.toml`

Configura o visual do painel Chainlit. A opção mais importante é `hide_cot = false` — faz aparecer os Steps internos ("🤖 Agent [CALENDARIO] · Latência: 1200ms") no painel. Em produção, não existe Chainlit. **Não vai ao Docker.**

---

### Arquivos de configuração

| Arquivo | Vai ao git? | Vai ao Docker? | Para que serve |
|---|---|---|---|
| `.env` | ❌ Não | Sim, via `env_file:` | Suas chaves reais |
| `.env.example` | ✅ Sim | Não | Template para novos devs |
| `docker-compose.yml` | ✅ Sim | É lido pelo Docker | Orquestra os containers |
| `Dockerfile` | ✅ Sim | Define a imagem | Receita da imagem do bot |
| `requirements.txt` | ✅ Sim | Sim, copiado e usado pelo pip | Dependências Python |
| `pyproject.toml` | ✅ Sim | Não diretamente | pytest, metadados do projeto |
| `debug/chainlit.toml` | ✅ Sim | ❌ Não | Visual do painel de debug |

---

## 9. Como uma mensagem é processada — pipeline completo

```
Usuário digita no WhatsApp
        │
        ▼
WAHA (container Docker) detecta a mensagem
Faz POST para: http://bot-rag:8000/webhook
        │
        ▼
src/main.py — FastAPI recebe o JSON bruto
        │
        ▼
src/application/handle_webhook.py
  │
  ├─ middleware/dev_guard.py valida:
  │    ✓ evento == "message"?
  │    ✓ não é fromMe?
  │    ✓ chat_id existe e não é grupo?
  │    ✓ DEV_MODE: sender está na whitelist?
  │    ✓ event_id não processado nos últimos 5min? (dedup Redis)
  │
  └─ Cria Mensagem(user_id, chat_id, body)
        │
        ▼
src/application/handle_message.py
  │
  ├─ 1. memory/redis_memory.get_estado_menu(user_id)
  │       → "MAIN" ou "SUB_CALENDARIO" etc.
  │
  ├─ 2. domain/menu.processar_mensagem(body, estado)
  │       Stateless. Sem Redis. Só recebe texto + estado.
  │       Decide: é menu principal? submenu? ou vai para o LLM?
  │
  ├─── SE tipo == "menu_principal" ou "submenu":
  │       memory/redis_memory.set_estado_menu(user_id, novo_estado)
  │       waha_service.enviar_mensagem(chat_id, texto_menu)
  │       FIM — sem chamar o Groq
  │
  └─── SE tipo == "llm":
         │
         ├─ domain/router.analisar(prompt, estado)
         │    Regex puro, sem I/O
         │    → Rota: CALENDARIO | EDITAL | CONTATOS | GERAL
         │
         ├─ memory/redis_memory.get_contexto(user_id)
         │    → {ultima_intencao: "EDITAL", nome: "João", ...}
         │
         ├─ agent/prompts.montar_prompt_enriquecido(prompt, rota, ctx)
         │    → "[CONTEXTO]\nÁrea: CALENDARIO\n..."
         │
         ├─ AgentState(user_id, rota, prompt_enriquecido, ...)
         │
         └─ agent/core.responder(state)
                │
                ├─ RunnableWithMessageHistory
                │    Carrega histórico Redis (sanitizado + sliding window)
                │
                ├─ Groq LLM recebe: system_prompt + histórico + mensagem
                │    Decide qual tool chamar e com qual query
                │
                ├─ tools/tool_calendario.py  (se rota == CALENDARIO)
                │    retriever.invoke(query normalizada)
                │    pgvector → k=4 chunks filtrados por source
                │    → "EVENTO: Matrícula | DATA: 03/02 | SEM: 2026.1"
                │
                ├─ tools/tool_edital.py  (se rota == EDITAL)
                │    → vagas, cotas, cronograma
                │
                ├─ tools/tool_contatos.py  (se rota == CONTATOS)
                │    → emails, telefones, responsáveis
                │
                ├─ Groq LLM sintetiza resposta final
                │    (máximo 3 parágrafos ou 6 itens)
                │
                └─ agent/validator.validar(state, output)
                       ✓ não é string interna do LangChain?
                       ✓ tem mais de 10 chars?
                       ✓ não está vazio?
                       → ValidationResult(valido, output_sanitizado)
                              │
         ┌─────────────────────┘
         │
         ├─ memory/redis_memory.set_contexto(user_id, {ultima_intencao: rota})
         ├─ infrastructure/observability.registrar_resposta(tokens, latência)
         └─ waha_service.enviar_mensagem(chat_id, resposta)
                │
                ▼
        WAHA envia ao WhatsApp do usuário
```

---

## 10. Pipeline de testes

### Três níveis de teste

```
tests/
├── unit/         → sem Docker, sem mocks, sem Redis — só Python puro
├── integration/  → com Redis e pgvector reais (docker-compose up db redis)
└── e2e/          → fluxo completo (Groq mockado)
```

### Testes unitários — rodam agora, sem nada instalado além do Python

```bash
pip install pytest pytest-asyncio pytest-cov
pytest tests/unit/ -v

# Com cobertura
pytest tests/unit/ --cov=src/domain --cov-report=term-missing
```

Os testes unitários testam **só a camada de domínio** — a única que não tem I/O. Não precisam de Docker, Redis, pgvector ou Groq.

**`tests/unit/test_menu.py`** — 9 testes de `domain/menu.py`:

```python
# Exemplos do que é testado:
resultado = processar_mensagem("oi", EstadoMenu.MAIN)
assert resultado["type"] == "menu_principal"           # saudação → menu

resultado = processar_mensagem("1", EstadoMenu.MAIN)
assert resultado["type"] == "submenu"                  # opção numérica
assert resultado["novo_estado"] == EstadoMenu.SUB_CALENDARIO

resultado = processar_mensagem("voltar", EstadoMenu.SUB_EDITAL)
assert resultado["novo_estado"] == EstadoMenu.MAIN     # volta do submenu

resultado = processar_mensagem("quando é a matrícula?", EstadoMenu.MAIN)
assert resultado["type"] == "llm"                      # texto livre → LLM
```

**`tests/unit/test_router.py`** — 12 testes de `domain/router.py`:

```python
assert analisar("data de matrícula", EstadoMenu.MAIN) == Rota.CALENDARIO
assert analisar("data de inscrição do PAES", EstadoMenu.MAIN) == Rota.EDITAL  # ambiguidade
assert analisar("email da PROG", EstadoMenu.MAIN) == Rota.CONTATOS
assert analisar("oi tudo bem", EstadoMenu.MAIN) == Rota.GERAL
assert analisar("qualquer coisa", EstadoMenu.SUB_EDITAL) == Rota.EDITAL  # forçado pelo submenu
```

**`tests/unit/test_validator.py`** — 8 testes de `agent/validator.py`:

```python
# Output vazio → inválido
r = validar(state, "")
assert r.valido == False

# String interna do LangChain → inválido, output substituído por mensagem amigável
r = validar(state, "Agent stopped due to max iterations.")
assert r.valido == False
assert "não encontrei" in r.output.lower()

# Output real → válido
r = validar(state, "A matrícula de veteranos ocorre de 03/02 a 07/02/2026.")
assert r.valido == True
```

### Adicionando novos testes

```python
# tests/unit/test_novo.py
from src.domain.menu import processar_mensagem
from src.domain.entities import EstadoMenu

def test_alias_edital_abre_submenu():
    r = processar_mensagem("vestibular", EstadoMenu.MAIN)
    assert r["type"] == "submenu"
    assert r["novo_estado"] == EstadoMenu.SUB_EDITAL

def test_opcao_no_submenu_vira_prompt_expandido():
    r = processar_mensagem("2", EstadoMenu.SUB_EDITAL)
    assert r["type"] == "llm"
    assert "documentos" in r["prompt"].lower()
```

### Testes de integração (estrutura preparada)

```bash
# Sobe só a infra
docker-compose up -d db redis

# Roda integration tests
pytest tests/integration/ -v
```

### Ciclo de desenvolvimento recomendado

```
1. Escreva o teste unitário primeiro (domínio puro)
2. Implemente a funcionalidade
3. pytest tests/unit/ -v  → deve passar
4. Suba a infra: docker-compose up -d db redis
5. pytest tests/integration/ -v  → com banco real
6. docker-compose up --build  → teste completo
7. curl /health + curl /banco/sources  → verifica PDFs
```

---

## 11. Painel de debug — Chainlit

```bash
# Instale (uma vez)
pip install chainlit tiktoken

# Rode da RAIZ do projeto
chainlit run debug/debug_chainlit.py --port 8001
# Acesse: http://localhost:8001
```

### O que o painel exibe

Cada mensagem mostra:
- A **rota detectada** pelo `domain/router.py` (ex: `CALENDARIO`)
- O **estado do menu** no momento (ex: `MAIN`)
- Um **Step interno** com latência e tokens estimados
- A **resposta final** do agente

### Comandos disponíveis no chat

| Comando | Descrição |
|---|---|
| `/ajuda` | Lista todos os comandos |
| `/status` | Modelo, LangSmith, HF_TOKEN, métricas da sessão |
| `/limpar` | Limpa histórico Redis + estado do menu do usuário de teste |
| `/diagnostico` | Quais PDFs foram ingeridos (debug do "Não encontrei") |
| `/modo agente` | Fluxo completo: menu → router → agente |
| `/modo direto` | Só o agente, sem menu/router |
| `/ingerir` | Força re-ingestão dos PDFs |
| `/exportar` | Baixa log completo da sessão em .txt |

### Diferença entre `/modo agente` e `/modo direto`

- **Modo agente**: passa pelo `domain/menu.py` e `domain/router.py` exatamente como em produção. Use para testar o comportamento real.
- **Modo direto**: vai direto ao `agent_core`, sem menu nem router. Use para testar respostas específicas isolando o agente.

---

## 12. LangSmith — rastreamento do agente

Rastreia automaticamente cada chamada do agente sem nenhuma mudança no código de negócio.

### Como ativar

```env
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=uema-bot
LANGCHAIN_TRACING_V2=true
```

### O que você vê no dashboard

- Qual tool foi chamada e com qual query exata
- Tokens de entrada/saída por step (cada chamada ao Groq)
- Latência total e por etapa
- Histórico de runs para comparar comportamentos
- Erros com stack trace completo e contexto

Acesse: [smith.langchain.com](https://smith.langchain.com) → projeto `uema-bot`.

O `agent/core.py` configura as variáveis de ambiente automaticamente no startup quando `settings.langsmith_ativo` for `True`.

---

## 13. Perguntas frequentes

**O bot responde "Não encontrei" para tudo.**

Os nomes dos PDFs não estão batendo com `SOURCE_*` nas tools. Confirme:

```bash
curl http://localhost:8000/banco/sources
# ou no Chainlit: /diagnostico
```

Os valores retornados devem ser IDÊNTICOS (case sensitive) às chaves em `src/rag/ingestor.py:PDF_CONFIG` e às constantes `SOURCE_*` em cada tool.

---

**O Groq retorna erro 429 (rate limit).**

O `providers/groq_provider.py` tem retry com backoff exponencial. Para reduzir chamadas: diminua `MAX_HISTORY_MESSAGES` (padrão: 6) e `AGENT_MAX_ITERATIONS` (padrão: 3) no `.env`.

---

**Erro 400 com "tool_use_failed" no log.**

O `memory/redis_memory.py` sanitiza automaticamente `tool_calls` órfãos no início de cada sessão. Se persistir, use `/limpar` no Chainlit ou adicione no `.env`:

```env
# Reinicia história de um usuário específico via endpoint ou código:
# from src.memory.redis_memory import clear_tudo
# clear_tudo("5598987654321")
```

---

**A ingestão está muito lenta.**

O modelo BAAI/bge-m3 (~1.3GB) é baixado na primeira vez. Configure `HF_TOKEN` no `.env` para evitar rate limit do HuggingFace Hub. O download ocorre só uma vez — depois fica em cache no container.

---

**Quero adicionar um novo PDF ao bot.**

1. Coloque o arquivo em `dados/`
2. Adicione a entrada em `src/rag/ingestor.py:PDF_CONFIG` com o nome exato
3. Crie `src/tools/tool_novo.py` com `SOURCE_NOVO = "nome-exato.pdf"`
4. Registre em `src/tools/__init__.py`
5. Reinicie o bot (a ingestão roda no startup) ou use `/ingerir` no Chainlit

---

**Quero adicionar uma nova área (ex: Suporte Técnico).**

1. `src/domain/entities.py` → adicione `SUPORTE = "SUPORTE"` em `Rota` e `SUB_SUPORTE` em `EstadoMenu`
2. `src/domain/router.py` → adicione o padrão regex em `_PADROES`
3. `src/domain/menu.py` → adicione texto e opções em `TEXTO_SUBMENU` e `OPCOES_SUBMENU`
4. `src/agent/prompts.py` → adicione contexto de rota em `_CONTEXTOS`
5. Crie e registre a tool correspondente

---

**Como trocar o modelo LLM?**

```env
# No .env
GROQ_MODEL=llama-3.1-8b-instant   # mais rápido, menos preciso
# ou
GROQ_MODEL=llama-3.3-70b-versatile  # mais preciso (padrão)
```

Modelos disponíveis no Groq free: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `gemma2-9b-it`.

---

**Como ver os logs e métricas?**

```bash
# Últimos 20 erros
curl http://localhost:8000/logs

# Últimas 50 respostas com tokens e latência
curl http://localhost:8000/metrics

# Verificar se Redis e agente estão ok
curl http://localhost:8000/health
```