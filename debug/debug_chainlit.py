"""
debug/debug_chainlit.py — Painel de Debug (v3)
===============================================

CORREÇÃO v2 — Problema de módulos não encontrados:
  O Chainlit roda na sua MÁQUINA LOCAL (fora do Docker).
  O Docker tem seus próprios Redis e pgvector nos IPs internos do container.
  Para o Chainlit funcionar na sua máquina, você precisa:

    1. Redis e pgvector expostos no localhost (já está no docker-compose.yml):
       db    → localhost:5433
       redis → localhost:6379

    2. Um arquivo .env.local na raiz com os hosts corretos:
       DATABASE_URL=postgresql+psycopg://postgres:senha@localhost:5433/vectordb
       REDIS_URL=redis://localhost:6379/0
       WAHA_BASE_URL=http://localhost:3000

    3. Dependências instaladas no venv local:
       pip install chainlit tiktoken langchain-groq langchain-postgres
       pip install langchain-huggingface langchain-community redis psycopg

  O Chainlit usa o .env.local automaticamente (via DOTENV_PATH abaixo).

USO:
  cd /caminho/para/meuBotRAG    ← sempre da RAIZ do projeto
  chainlit run debug/debug_chainlit.py --port 8001

COMANDOS NO CHAT:
  /ajuda · /status · /limpar · /diagnostico
  /modo agente · /modo direto · /ingerir · /exportar
"""
from __future__ import annotations

import os
import sys
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# ── Verificação de versão ─────────────────────────────────────────────────────
# Chainlit suporta 3.9 a 3.12. Python 3.12 funciona normalmente no Windows.
if sys.version_info >= (3, 13):
    print("❌  Python 3.13+ não suportado pelo Chainlit.")
    print("    Use 3.11 ou 3.12.")
    sys.exit(1)

# ── Resolve raiz do projeto ───────────────────────────────────────────────────
# Funciona rodando de debug/ ou da raiz
_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parent if _AQUI.name == "debug" else _AQUI

if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

# ── .env local para Chainlit (fora do Docker) ─────────────────────────────────
# Carrega .env.local se existir, senão usa .env padrão
# .env.local deve ter hosts de localhost em vez de nomes de serviço Docker
_ENV_LOCAL  = _RAIZ / ".env.local"
_ENV_PADRAO = _RAIZ / ".env"
_ENV_FILE   = str(_ENV_LOCAL if _ENV_LOCAL.exists() else _ENV_PADRAO)
os.environ["ENV_FILE_PATH"] = _ENV_FILE   # lido pelo settings.py se configurado

# Sobrescreve o env_file do pydantic-settings antes de importar settings
# (funciona porque settings usa @lru_cache — ainda não foi chamado)
if _ENV_LOCAL.exists():
    # Força o pydantic-settings a usar .env.local
    # Técnica: define as variáveis no ambiente antes de importar settings
    from dotenv import dotenv_values
    _local_vars = dotenv_values(_ENV_LOCAL)
    for k, v in _local_vars.items():
        if v is not None and k not in os.environ:
            os.environ[k] = v
    print(f"🔧 Chainlit usando: {_ENV_LOCAL}")
else:
    print(f"🔧 Chainlit usando: {_ENV_PADRAO}")
    print("   💡 Crie .env.local com hosts de localhost para isolamento do Docker")

# ── Desativa o ChainlitDataLayer (banco interno do Chainlit) ─────────────────
# O Chainlit por padrão tenta conectar num banco próprio (asyncpg) para salvar
# histórico de sessões. No Windows/local ele não tem esse banco configurado,
# causando dezenas de "ConnectionRefusedError: WinError 1225" nos logs.
#
# Solução: define CHAINLIT_AUTH_SECRET como string vazia antes de importar
# chainlit. Com isso, o Chainlit opera sem persistência — perfeito para debug.
#
# Isso NÃO afeta o funcionamento do agente, Redis ou pgvector do projeto.
os.environ.setdefault("CHAINLIT_AUTH_SECRET", "")

import chainlit as cl

# ── Imports dos módulos de produção ───────────────────────────────────────────
_MODULOS_OK  = True
_ERRO_IMPORT = None
try:
    from src.infrastructure.settings     import settings
    from src.infrastructure.observability import obs
    from src.agent.core                  import agent_core
    from src.agent.state                 import AgentState
    from src.agent.prompts               import montar_prompt_enriquecido
    from src.domain.menu                 import processar_mensagem
    from src.domain.router               import analisar
    from src.domain.entities             import EstadoMenu
    from src.memory.redis_memory         import (
        get_estado_menu, set_estado_menu, clear_estado_menu,
        get_contexto, set_contexto, clear_tudo,
    )
    from src.rag.ingestor                import Ingestor, PDF_CONFIG
    from src.rag.vector_store            import diagnosticar as vs_diagnosticar
    from src.tools                       import get_tools_ativas
except ImportError as e:
    _MODULOS_OK  = False
    _ERRO_IMPORT = str(e)
    print(f"\n❌ Erro ao importar módulos: {e}")
    print("\nVerifique:")
    print("  1. Você está na RAIZ do projeto (não dentro de debug/)")
    print("  2. O venv está ativo com todas as dependências:")
    print("     pip install -r requirements.txt chainlit tiktoken")
    print("  3. O .env.local tem DATABASE_URL e REDIS_URL com localhost")
    print("     (não nomes de serviço Docker como 'db' ou 'redis')")

# ── Contador de tokens ─────────────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _tokens(t: str) -> int:
        return len(_enc.encode(str(t)))
except Exception:
    def _tokens(t: str) -> int:
        return len(str(t)) // 4

logging.basicConfig(level=logging.WARNING)

_DEBUG_USER = "debug_chainlit"


# =============================================================================
# Estado da sessão
# =============================================================================

def _novo_estado() -> dict:
    return {
        "modo": "agente",
        "iniciado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "msgs": 0, "tokens": 0, "latencia_total": 0, "log": [],
    }


# =============================================================================
# on_chat_start
# =============================================================================

@cl.on_chat_start
async def on_start():
    cl.user_session.set("s", _novo_estado())

    if not _MODULOS_OK:
        env_dica = (
            "`.env.local` encontrado ✅"
            if (_RAIZ / ".env.local").exists()
            else "`.env.local` **não encontrado** — crie com hosts de localhost"
        )
        await cl.Message(content=(
            f"⚠️ **Módulos não carregados**\n\n```\n{_ERRO_IMPORT}\n```\n\n"
            f"**Arquivo de configuração:** {env_dica}\n\n"
            "**Checklist:**\n"
            "```bash\n"
            "# 1. Rode da raiz do projeto\n"
            "cd /caminho/para/meuBotRAG\n\n"
            "# 2. Ative o venv com as dependências\n"
            "source .venv/bin/activate\n"
            "pip install chainlit tiktoken langchain-groq langchain-postgres\n"
            "pip install langchain-huggingface langchain-community redis psycopg\n\n"
            "# 3. Crie o .env.local (hosts de localhost, não Docker)\n"
            "cp .env .env.local\n"
            "# Edite .env.local:\n"
            "# DATABASE_URL=postgresql+psycopg://postgres:senha@localhost:5433/vectordb\n"
            "# REDIS_URL=redis://localhost:6379/0\n\n"
            "# 4. Rode novamente\n"
            "chainlit run debug/debug_chainlit.py --port 8001\n"
            "```"
        )).send()
        return

    # Inicializa o agente se necessário
    if not agent_core._agent_with_history:
        async with cl.Step(name="🔧 Inicializando agente") as step:
            try:
                ingestor = Ingestor()
                await asyncio.to_thread(ingestor.ingerir_se_necessario)
                tools = get_tools_ativas()
                await asyncio.to_thread(agent_core.inicializar, tools)
                step.output = f"✅ {len(tools)} tools | {settings.GROQ_MODEL}"
            except Exception as e:
                step.output = f"❌ {e}"

    ls = (
        f"✅ [ver traces](https://smith.langchain.com) → `{settings.LANGCHAIN_PROJECT}`"
        if settings.langsmith_ativo else "❌ desativado"
    )
    env_msg = "`.env.local`" if (_RAIZ / ".env.local").exists() else "`.env`"

    await cl.Message(content=(
        "## 🎓 Debug — Agente UEMA\n\n"
        f"**Modo:** `agente` &nbsp;|&nbsp; **Modelo:** `{settings.GROQ_MODEL}`\n\n"
        f"**Config:** {env_msg} &nbsp;|&nbsp; "
        f"**DB:** `{settings.DATABASE_URL.split('@')[-1]}`\n\n"
        f"**LangSmith:** {ls}\n"
        f"**HF_TOKEN:** {'✅' if settings.HF_TOKEN else '⚠️ ausente'}\n\n"
        "---\nDigite ou use: "
        "`/ajuda` · `/status` · `/limpar` · `/diagnostico` · `/modo direto` · `/ingerir`"
    )).send()


# =============================================================================
# on_message
# =============================================================================

@cl.on_message
async def on_message(message: cl.Message):
    texto = message.content.strip()
    s     = cl.user_session.get("s") or _novo_estado()

    if texto.startswith("/"):
        await _cmd(texto, s)
    elif not _MODULOS_OK:
        await cl.Message(content="⚠️ Módulos não carregados. Veja o erro no início.").send()
    elif s["modo"] == "agente":
        await _modo_agente(texto, s)
    else:
        await _modo_direto(texto, s)

    cl.user_session.set("s", s)


# =============================================================================
# Modos de resposta
# =============================================================================

async def _modo_agente(texto: str, s: dict) -> None:
    t0        = time.time()
    modo_menu = get_estado_menu(_DEBUG_USER)
    resultado = processar_mensagem(texto, modo_menu)

    if resultado["type"] in ("menu_principal", "submenu"):
        set_estado_menu(_DEBUG_USER, resultado["novo_estado"])
        await cl.Message(content=resultado["content"], author="📋 Menu").send()
        _log(s, texto, resultado["content"], 0, "menu")
        return

    novo = resultado["novo_estado"]
    if novo == EstadoMenu.MAIN:
        clear_estado_menu(_DEBUG_USER)
    else:
        set_estado_menu(_DEBUG_USER, novo)

    prompt_base  = resultado["prompt"] or texto
    rota         = analisar(prompt_base, modo_menu)
    ctx          = get_contexto(_DEBUG_USER)
    prompt_final = montar_prompt_enriquecido(prompt_base, rota, ctx)

    await cl.Message(
        content=f"`🔍 Rota: {rota.value}` · `Menu: {modo_menu.value}`",
        author="Router"
    ).send()

    state = AgentState(
        user_id=_DEBUG_USER, session_id=_DEBUG_USER,
        mensagem_original=texto, chat_id="debug",
        rota=rota, modo_menu=modo_menu,
        prompt_enriquecido=prompt_final, contexto_usuario=ctx,
        max_iteracoes=settings.AGENT_MAX_ITERATIONS,
    )

    async with cl.Step(name=f"🤖 Agent [{rota.value}]") as step:
        resp     = await asyncio.to_thread(agent_core.responder, state)
        latencia = int((time.time() - t0) * 1000)
        toks     = _tokens(texto) + _tokens(resp.conteudo)
        step.output = f"**{latencia}ms** · ~{toks} tokens · {'✅' if resp.sucesso else '❌'}"

    set_contexto(_DEBUG_USER, {"ultima_intencao": rota.value})
    await cl.Message(content=resp.conteudo).send()
    _log(s, texto, resp.conteudo, latencia, f"agente/{rota.value}")


async def _modo_direto(texto: str, s: dict) -> None:
    t0    = time.time()
    state = AgentState(
        user_id=_DEBUG_USER, session_id=_DEBUG_USER,
        mensagem_original=texto, chat_id="debug",
    )
    async with cl.Step(name="🤖 Agent [direto]") as step:
        resp     = await asyncio.to_thread(agent_core.responder, state)
        latencia = int((time.time() - t0) * 1000)
        step.output = f"**{latencia}ms** · {'✅' if resp.sucesso else '❌'}"
    await cl.Message(content=resp.conteudo).send()
    _log(s, texto, resp.conteudo, latencia, "direto")


def _log(s: dict, p: str, r: str, lat: int, modo: str):
    s["msgs"] += 1; s["tokens"] += _tokens(p) + _tokens(r); s["latencia_total"] += lat
    s["log"].append({"ts": datetime.now().strftime("%H:%M:%S"), "modo": modo,
                     "latencia": lat, "pergunta": p, "resposta": r})


# =============================================================================
# Comandos
# =============================================================================

async def _cmd(texto: str, s: dict) -> None:
    partes = texto.lower().split()
    cmd    = partes[0]

    if cmd == "/ajuda":
        await cl.Message(content=(
            "## Comandos\n\n"
            "| Comando | Descrição |\n|---|---|\n"
            "| `/ajuda` | Esta mensagem |\n"
            "| `/status` | Config, LangSmith, HF_TOKEN, métricas |\n"
            "| `/limpar` | Limpa histórico Redis + estado do menu |\n"
            "| `/diagnostico` | Sources no banco vetorial |\n"
            "| `/modo agente` | Fluxo completo: menu → router → agente |\n"
            "| `/modo direto` | Só o agente, sem menu/router |\n"
            "| `/ingerir` | Força re-ingestão dos PDFs |\n"
            "| `/exportar` | Baixa log da sessão em .txt |\n"
        )).send()

    elif cmd == "/status":
        msgs = s["msgs"]
        lat  = (s["latencia_total"] // msgs) if msgs else 0
        ls   = (f"✅ `{settings.LANGCHAIN_PROJECT}`" if settings.langsmith_ativo else "❌")
        db_host = settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else "?"
        await cl.Message(content=(
            f"## Status\n\n"
            f"**Modo:** `{s['modo']}` · **Modelo:** `{settings.GROQ_MODEL}`\n"
            f"**Agente:** {'✅' if agent_core._agent_with_history else '❌'}\n"
            f"**DB host:** `{db_host}`\n"
            f"**Redis:** `{settings.REDIS_URL}`\n"
            f"**LangSmith:** {ls} · **HF_TOKEN:** {'✅' if settings.HF_TOKEN else '⚠️'}\n\n"
            f"**Sessão:** {msgs} msgs · ~{s['tokens']} tokens · lat. média {lat}ms\n"
            f"**Iniciado:** {s['iniciado_em']}\n"
        )).send()

    elif cmd == "/limpar":
        if _MODULOS_OK:
            clear_tudo(_DEBUG_USER)
            await cl.Message(content="🗑️ Histórico + estado do menu limpos.").send()

    elif cmd == "/diagnostico":
        if not _MODULOS_OK:
            await cl.Message(content="⚠️ Módulos não carregados.").send()
            return
        async with cl.Step(name="🔍 Verificando banco") as step:
            sources = await asyncio.to_thread(vs_diagnosticar)
            step.output = str(sources)
        esperados = set(PDF_CONFIG.keys())
        faltam    = esperados - sources
        linhas    = ["### Sources no banco\n"]
        for src in sorted(sources):
            linhas.append(f"- `{src}` {'✅' if src in esperados else '⚠️ não esperado'}")
        if faltam:
            linhas.append(f"\n❌ **Faltam:** {', '.join(f'`{f}`' for f in faltam)}")
            linhas.append("\n💡 Use `/ingerir` para processar os PDFs ausentes.")
        else:
            linhas.append("\n✅ Todos os arquivos esperados estão no banco.")
        await cl.Message(content="\n".join(linhas)).send()

    elif cmd == "/modo":
        novo = partes[1] if len(partes) > 1 else ""
        if novo in ("agente", "direto"):
            s["modo"] = novo
            await cl.Message(content=f"✅ Modo: `{novo}`").send()
        else:
            await cl.Message(content="❌ Use `/modo agente` ou `/modo direto`.").send()

    elif cmd == "/ingerir":
        if not _MODULOS_OK:
            await cl.Message(content="⚠️ Módulos não carregados.").send()
            return
        async with cl.Step(name="📥 Re-ingerindo PDFs") as step:
            try:
                await asyncio.to_thread(Ingestor().ingerir_tudo)
                step.output = "✅ Concluído."
            except Exception as e:
                step.output = f"❌ {e}"
        await cl.Message(content="✅ Re-ingestão concluída. Use `/diagnostico` para confirmar.").send()

    elif cmd == "/exportar":
        if not s["log"]:
            await cl.Message(content="Nenhuma mensagem nesta sessão.").send()
            return
        linhas = [
            f"# Log de Debug — Agente UEMA",
            f"# Gerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Mensagens: {s['msgs']} · Tokens ~{s['tokens']}",
            "",
        ]
        for i, h in enumerate(s["log"], 1):
            linhas += [f"─── [{i}] {h['ts']} | {h['modo']} | {h['latencia']}ms",
                       f">>> {h['pergunta']}", f"<<< {h['resposta']}", ""]
        await cl.Message(
            content="📄 Log:",
            elements=[cl.File(
                name=f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                content="\n".join(linhas).encode("utf-8"), mime="text/plain",
            )]
        ).send()
    else:
        await cl.Message(content=f"❓ Desconhecido: `{cmd}`. Use `/ajuda`.").send()