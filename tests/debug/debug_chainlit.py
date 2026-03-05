"""
debug/debug_chainlit.py — Painel de Debug (v3 — Clean Architecture)
====================================================================

O QUE MUDOU vs v2:
─────────────────────
  REMOVIDO:
    - Imports de AgentState, montar_prompt_enriquecido (agent/prompts.py)
    - Import de domain/router.py (analisar) — agora é interno ao semantic_router
    - Import de rag/vector_store.py (diagnosticar) — substituído por Redis
    - Referências a settings.GROQ_MODEL, settings.LANGCHAIN_PROJECT
    - Referências a DATABASE_URL (pgvector eliminado)
    - Modo "direto" antigo (usava AgentState directamente)
    - Verificação agent_core._agent_with_history

  ADICIONADO:
    - Status do Redis Stack (módulos RedisSearch + RedisJSON)
    - settings.GEMINI_MODEL nos logs
    - /fatos no painel de diagnóstico (long-term memory)
    - /memoria no painel de diagnóstico (working memory)
    - Comando /fatos para ver fatos do utilizador de debug
    - Comando /extracao para forçar extração de fatos
    - Modo "direto" redesenhado para nova API do AgentCore
    - Verificação agent_core._inicializado

  MANTIDO:
    - Estrutura de comandos (/ajuda, /status, /limpar, /ingerir, /exportar)
    - Modo agente (adaptado à nova assinatura do AgentCore)
    - Sistema de log de sessão

COMO CORRER (igual ao anterior):
──────────────────────────────────
  # Sempre da RAIZ do projecto:
  cd /caminho/para/meuBotRAG

  # Activa o venv com as dependências:
  source .venv/bin/activate

  # Instala dependências extra para debug local:
  pip install chainlit tiktoken google-genai redis[hiredis]

  # Redis Stack deve estar a correr (via Docker ou local):
  # docker run -p 6379:6379 redis/redis-stack:latest

  # Cria .env.local com hosts de localhost:
  # REDIS_URL=redis://localhost:6379/0
  # GEMINI_API_KEY=...
  # (sem DATABASE_URL — pgvector eliminado)

  # Corre o Chainlit:
  chainlit run debug/debug_chainlit.py --port 8001

NOTA SOBRE O DOCKER:
──────────────────────
  Este script corre na máquina LOCAL (fora do Docker).
  O Redis Stack no Docker expõe a porta 6379 → usas REDIS_URL=redis://localhost:6379/0
  O bot FastAPI no Docker está em localhost:8000
  Os PDFs ingeridos no Docker estão no Redis → o Chainlit acede ao mesmo Redis
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Verificação de versão ─────────────────────────────────────────────────────
if sys.version_info >= (3, 13):
    print("❌  Python 3.13+ não suportado pelo Chainlit. Use 3.11 ou 3.12.")
    sys.exit(1)

# ── Resolve raiz do projecto ──────────────────────────────────────────────────
_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parent if _AQUI.name == "debug" else _AQUI

if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

# ── Carrega .env.local se existir, senão usa .env ─────────────────────────────
_ENV_LOCAL  = _RAIZ / ".env.local"
_ENV_PADRAO = _RAIZ / ".env"

if _ENV_LOCAL.exists():
    from dotenv import dotenv_values
    for k, v in dotenv_values(_ENV_LOCAL).items():
        if v is not None and k not in os.environ:
            os.environ[k] = v
    print(f"🔧 Chainlit usando: {_ENV_LOCAL}")
else:
    print(f"🔧 Chainlit usando: {_ENV_PADRAO}")
    print("   💡 Cria .env.local com REDIS_URL=redis://localhost:6379/0 e GEMINI_API_KEY=...")

# ── Desactiva o ChainlitDataLayer ─────────────────────────────────────────────
os.environ.setdefault("CHAINLIT_AUTH_SECRET", "")

import chainlit as cl

# ── Imports dos módulos de produção ───────────────────────────────────────────
_MODULOS_OK  = True
_ERRO_IMPORT = None

try:
    from src.infrastructure.settings    import settings
    from src.infrastructure.observability import obs
    from src.infrastructure.redis_client import redis_ok, inicializar_indices
    from src.agent.core                  import agent_core
    from src.domain.menu                 import processar_mensagem
    from src.domain.entities             import EstadoMenu
    from src.memory.redis_memory         import (
        get_estado_menu, set_estado_menu, clear_estado_menu, clear_tudo,
    )
    # Módulos novos (v3)
    from src.memory.working_memory       import (
        get_historico_compactado, get_sinais, limpar_sessao,
    )
    from src.memory.long_term_memory     import listar_todos_fatos
    from src.memory.memory_extractor     import forcar_extracao, testar_extracao
    from src.domain.semantic_router      import testar_roteamento, listar_tools_registadas
    from src.rag.ingestor                import Ingestor, PDF_CONFIG
    from src.tools                       import get_tools_ativas

except ImportError as e:
    _MODULOS_OK  = False
    _ERRO_IMPORT = str(e)
    print(f"\n❌ Erro ao importar módulos: {e}")
    print("\nVerifique:")
    print("  1. Estás na RAIZ do projecto (não dentro de debug/)")
    print("  2. O venv está activo:")
    print("     pip install -r requirements.txt chainlit tiktoken google-genai")
    print("  3. O .env.local tem REDIS_URL=redis://localhost:6379/0")
    print("  4. O Redis Stack está a correr: docker run -p 6379:6379 redis/redis-stack:latest")

# ── Estimador de tokens (sem tiktoken obrigatório) ────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _tokens(t: str) -> int:
        return len(_enc.encode(str(t)))
except Exception:
    def _tokens(t: str) -> int:
        return len(str(t)) // 4   # Estimativa conservadora para português

logging.basicConfig(level=logging.WARNING)

_DEBUG_USER    = "debug_chainlit"
_DEBUG_SESSION = "debug_chainlit"


# =============================================================================
# Estado da sessão Chainlit
# =============================================================================

def _novo_estado() -> dict:
    return {
        "modo":        "agente",
        "iniciado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "msgs":        0,
        "tokens":      0,
        "latencia_total": 0,
        "log":         [],
    }


# =============================================================================
# on_chat_start
# =============================================================================

@cl.on_chat_start
async def on_start():
    cl.user_session.set("s", _novo_estado())

    # ── Mostra erro de import se aconteceu ────────────────────────────────────
    if not _MODULOS_OK:
        await cl.Message(content=(
            f"⚠️ **Módulos não carregados**\n\n```\n{_ERRO_IMPORT}\n```\n\n"
            "**Checklist:**\n"
            "```bash\n"
            "cd /caminho/para/meuBotRAG\n"
            "source .venv/bin/activate\n"
            "pip install -r requirements.txt chainlit tiktoken google-genai\n\n"
            "# Cria .env.local:\n"
            "echo 'REDIS_URL=redis://localhost:6379/0' > .env.local\n"
            "echo 'GEMINI_API_KEY=tua_chave_aqui' >> .env.local\n\n"
            "# Inicia Redis Stack (noutra terminal):\n"
            "docker run -p 6379:6379 redis/redis-stack:latest\n\n"
            "chainlit run debug/debug_chainlit.py --port 8001\n"
            "```"
        )).send()
        return

    # ── Inicializa agente se necessário ──────────────────────────────────────
    if not agent_core._inicializado:
        async with cl.Step(name="🔧 Inicializando agente v3") as step:
            try:
                # Cria índices Redis Stack
                await asyncio.to_thread(inicializar_indices)

                # Ingere PDFs se necessário
                ingestor = Ingestor()
                await asyncio.to_thread(ingestor.ingerir_se_necessario)

                # Inicializa AgentCore com tools
                tools = get_tools_ativas()
                await asyncio.to_thread(agent_core.inicializar, tools)

                step.output = f"✅ {len(tools)} tools | {settings.GEMINI_MODEL} | Redis Stack"
            except Exception as e:
                step.output = f"❌ {e}"
                await cl.Message(content=f"⚠️ Erro ao inicializar: {e}").send()
                return

    # ── Mensagem de boas-vindas ───────────────────────────────────────────────
    redis_status = "✅" if redis_ok() else "❌"
    env_msg      = "`.env.local`" if _ENV_LOCAL.exists() else "`.env`"
    tools_reg    = listar_tools_registadas() if _MODULOS_OK else []

    await cl.Message(content=(
        "## 🎓 Debug — Agente UEMA v3\n\n"
        f"**Config:** {env_msg} &nbsp;|&nbsp; "
        f"**Modelo:** `{settings.GEMINI_MODEL}`\n\n"
        f"**Redis Stack:** {redis_status} `{settings.REDIS_URL}`\n"
        f"**Tools registadas:** {', '.join(f'`{t}`' for t in tools_reg) or '(nenhuma)'}\n"
        f"**pgvector:** ❌ eliminado (Redis Stack faz tudo)\n\n"
        "---\n"
        "**Comandos:** `/ajuda` · `/status` · `/limpar` · `/diagnostico` "
        "· `/fatos` · `/extracao` · `/router [query]` · `/ingerir` · `/exportar`"
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
        await cl.Message(content="⚠️ Módulos não carregados. Vê o erro no início.").send()
    elif s["modo"] == "agente":
        await _modo_agente(texto, s)
    else:
        await _modo_direto(texto, s)

    cl.user_session.set("s", s)


# =============================================================================
# Modos de resposta
# =============================================================================

async def _modo_agente(texto: str, s: dict) -> None:
    """
    Fluxo completo: menu → AgentCore v3 (Gemini + Redis).
    Mostra métricas de cada passo no painel.
    """
    t0        = time.time()
    modo_menu = get_estado_menu(_DEBUG_USER)
    resultado = processar_mensagem(texto, modo_menu)

    # ── Navegação de menu directa ─────────────────────────────────────────────
    if resultado["type"] in ("menu_principal", "submenu"):
        set_estado_menu(_DEBUG_USER, resultado["novo_estado"])
        await cl.Message(content=resultado["content"], author="📋 Menu").send()
        _log(s, texto, resultado["content"], 0, "menu")
        return

    # ── Actualiza estado do menu ──────────────────────────────────────────────
    novo = resultado["novo_estado"]
    if novo != modo_menu:
        if novo == EstadoMenu.MAIN:
            clear_estado_menu(_DEBUG_USER)
        else:
            set_estado_menu(_DEBUG_USER, novo)

    texto_para_agente = resultado.get("prompt") or texto

    # ── Mostra info de roteamento antes de chamar o agente ────────────────────
    await cl.Message(
        content=f"`🗺️ Estado menu: {modo_menu.value}` · `Texto: {texto_para_agente[:50]}`",
        author="Pre-Pipeline",
    ).send()

    # ── Chama o novo AgentCore v3 ─────────────────────────────────────────────
    async with cl.Step(name="🤖 AgentCore v3 [Gemini + Redis]") as step:
        resp     = await asyncio.to_thread(
            agent_core.responder,
            user_id=_DEBUG_USER,
            session_id=_DEBUG_SESSION,
            mensagem=texto_para_agente,
            estado_menu=modo_menu,
        )
        latencia = int((time.time() - t0) * 1000)
        toks_in  = resp.tokens_entrada
        toks_out = resp.tokens_saida

        step.output = (
            f"**{latencia}ms** · "
            f"in={toks_in} out={toks_out} total={toks_in + toks_out} tokens · "
            f"rota={resp.rota.value} · "
            f"{'✅' if resp.sucesso else '❌'}"
        )

    await cl.Message(content=resp.conteudo).send()
    _log(s, texto, resp.conteudo, latencia, f"agente/{resp.rota.value}")


async def _modo_direto(texto: str, s: dict) -> None:
    """
    Modo direto (sem menu): chama o AgentCore directamente com EstadoMenu.MAIN.
    Útil para testar o pipeline RAG puro sem lógica de menu.
    """
    t0 = time.time()
    async with cl.Step(name="🤖 AgentCore [modo direto]") as step:
        resp     = await asyncio.to_thread(
            agent_core.responder,
            user_id=_DEBUG_USER,
            session_id=_DEBUG_SESSION,
            mensagem=texto,
            estado_menu=EstadoMenu.MAIN,
        )
        latencia = int((time.time() - t0) * 1000)
        step.output = f"**{latencia}ms** · rota={resp.rota.value} · {'✅' if resp.sucesso else '❌'}"

    await cl.Message(content=resp.conteudo).send()
    _log(s, texto, resp.conteudo, latencia, f"direto/{resp.rota.value}")


def _log(s: dict, p: str, r: str, lat: int, modo: str) -> None:
    s["msgs"]          += 1
    s["tokens"]        += _tokens(p) + _tokens(r)
    s["latencia_total"] += lat
    s["log"].append({
        "ts":       datetime.now().strftime("%H:%M:%S"),
        "modo":     modo,
        "latencia": lat,
        "pergunta": p,
        "resposta": r,
    })


# =============================================================================
# Comandos /cmd
# =============================================================================

async def _cmd(texto: str, s: dict) -> None:
    partes = texto.lower().split()
    cmd    = partes[0]

    # ── /ajuda ────────────────────────────────────────────────────────────────
    if cmd == "/ajuda":
        await cl.Message(content=(
            "## Comandos disponíveis\n\n"
            "| Comando | Descrição |\n|---|---|\n"
            "| `/ajuda` | Esta mensagem |\n"
            "| `/status` | Estado do sistema (Redis, Gemini, tools) |\n"
            "| `/limpar` | Limpa working memory + estado menu da sessão |\n"
            "| `/diagnostico` | Verifica sources no Redis |\n"
            "| `/fatos` | Lista fatos long-term do utilizador de debug |\n"
            "| `/extracao` | Força extração de fatos da conversa atual |\n"
            "| `/router <query>` | Testa o semantic router para uma query |\n"
            "| `/modo agente` | Fluxo completo: menu → AgentCore v3 |\n"
            "| `/modo direto` | Só o AgentCore, sem lógica de menu |\n"
            "| `/ingerir` | Força re-ingestão dos PDFs no Redis |\n"
            "| `/exportar` | Exporta log da sessão em .txt |\n"
        )).send()

    # ── /status ───────────────────────────────────────────────────────────────
    elif cmd == "/status":
        msgs     = s["msgs"]
        lat_med  = (s["latencia_total"] // msgs) if msgs else 0
        tools    = listar_tools_registadas() if _MODULOS_OK else []

        await cl.Message(content=(
            f"## Status — Bot UEMA v3\n\n"
            f"**Modo:** `{s['modo']}` &nbsp;|&nbsp; "
            f"**Modelo:** `{settings.GEMINI_MODEL}`\n\n"
            f"**AgentCore:** {'✅ pronto' if agent_core._inicializado else '❌ não inicializado'}\n"
            f"**Redis Stack:** {'✅' if redis_ok() else '❌'} `{settings.REDIS_URL}`\n"
            f"**pgvector:** ❌ eliminado\n"
            f"**HF_TOKEN:** {'✅' if settings.HF_TOKEN else '⚠️ ausente (download anónimo)'}\n\n"
            f"**Tools registadas ({len(tools)}):** {', '.join(f'`{t}`' for t in tools)}\n\n"
            f"**Sessão:** {msgs} msgs · ~{s['tokens']} tokens · lat. média {lat_med}ms\n"
            f"**Iniciado:** {s['iniciado_em']}\n"
        )).send()

    # ── /limpar ───────────────────────────────────────────────────────────────
    elif cmd == "/limpar":
        if _MODULOS_OK:
            # Limpa working memory (nova) + estado menu + contexto antigo
            await asyncio.to_thread(limpar_sessao, _DEBUG_SESSION)
            clear_tudo(_DEBUG_USER)
            await cl.Message(content="🗑️ Working memory + estado menu + fatos de sessão limpos.").send()

    # ── /diagnostico ──────────────────────────────────────────────────────────
    elif cmd == "/diagnostico":
        if not _MODULOS_OK:
            await cl.Message(content="⚠️ Módulos não carregados.").send()
            return

        async with cl.Step(name="🔍 Verificando Redis Stack") as step:
            ingestor = Ingestor()
            sources  = await asyncio.to_thread(ingestor.diagnosticar)
            step.output = f"Sources encontrados: {sources}"

        esperados = set(PDF_CONFIG.keys())
        faltam    = esperados - sources
        linhas    = ["### Sources no Redis\n"]
        for src in sorted(sources):
            linhas.append(f"- `{src}` {'✅' if src in esperados else '⚠️ não esperado'}")
        if faltam:
            linhas.append(f"\n❌ **Faltam:** {', '.join(f'`{f}`' for f in faltam)}")
            linhas.append("💡 Use `/ingerir` para processar os PDFs em falta.")
        else:
            linhas.append("\n✅ Todos os ficheiros esperados estão no Redis.")

        # Working memory actual
        hist  = await asyncio.to_thread(get_historico_compactado, _DEBUG_SESSION)
        sinais = await asyncio.to_thread(get_sinais, _DEBUG_SESSION)
        linhas.append(f"\n### Working Memory\n- Turns: {hist.turns_incluidos} | Chars: {hist.total_chars}")
        if sinais:
            for k, v in sinais.items():
                linhas.append(f"- {k}: `{v}`")

        await cl.Message(content="\n".join(linhas)).send()

    # ── /fatos ────────────────────────────────────────────────────────────────
    elif cmd == "/fatos":
        if not _MODULOS_OK:
            await cl.Message(content="⚠️ Módulos não carregados.").send()
            return

        fatos = await asyncio.to_thread(listar_todos_fatos, _DEBUG_USER)
        if not fatos:
            await cl.Message(content="ℹ️ Sem fatos long-term para o utilizador de debug.\nFaz algumas perguntas e usa `/extracao` para extrair.").send()
        else:
            linhas = [f"### Fatos Long-Term de `{_DEBUG_USER}` ({len(fatos)} total)\n"]
            for i, f in enumerate(fatos, 1):
                linhas.append(f"{i}. {f}")
            await cl.Message(content="\n".join(linhas)).send()

    # ── /extracao ─────────────────────────────────────────────────────────────
    elif cmd == "/extracao":
        if not _MODULOS_OK:
            await cl.Message(content="⚠️ Módulos não carregados.").send()
            return

        async with cl.Step(name="🧠 Forçando extração de fatos") as step:
            guardados = await asyncio.to_thread(forcar_extracao, _DEBUG_USER, _DEBUG_SESSION)
            step.output = f"✅ {guardados} novos fatos guardados"

        await cl.Message(content=f"🧠 Extração concluída: {guardados} novo(s) fato(s).\nUsa `/fatos` para ver.").send()

    # ── /router <query> ───────────────────────────────────────────────────────
    elif cmd == "/router":
        query = " ".join(partes[1:]) if len(partes) > 1 else "matrícula veteranos"
        if not _MODULOS_OK:
            await cl.Message(content="⚠️ Módulos não carregados.").send()
            return

        async with cl.Step(name=f"🗺️ Testando router: '{query}'") as step:
            resultado = await asyncio.to_thread(testar_roteamento, query)
            step.output = str(resultado)

        linhas = [f"### Resultado do Semantic Router\n\n**Query:** `{query}`\n"]
        for r in resultado:
            linhas.append(
                f"- **{r.get('tool', '?')}** → rota=`{r.get('rota', '?')}` "
                f"| score=`{r.get('score', 0):.3f}` "
                f"| confiança=`{r.get('confianca', '?')}`"
            )
        await cl.Message(content="\n".join(linhas)).send()

    # ── /modo ─────────────────────────────────────────────────────────────────
    elif cmd == "/modo":
        novo = partes[1] if len(partes) > 1 else ""
        if novo in ("agente", "direto"):
            s["modo"] = novo
            await cl.Message(content=f"✅ Modo: `{novo}`").send()
        else:
            await cl.Message(content="❌ Usa `/modo agente` ou `/modo direto`.").send()

    # ── /ingerir ──────────────────────────────────────────────────────────────
    elif cmd == "/ingerir":
        if not _MODULOS_OK:
            await cl.Message(content="⚠️ Módulos não carregados.").send()
            return
        async with cl.Step(name="📥 Re-ingerindo PDFs no Redis") as step:
            try:
                await asyncio.to_thread(Ingestor().ingerir_tudo)
                step.output = "✅ Concluído."
            except Exception as e:
                step.output = f"❌ {e}"
        await cl.Message(content="✅ Re-ingestão concluída no Redis. Usa `/diagnostico` para confirmar.").send()

    # ── /exportar ─────────────────────────────────────────────────────────────
    elif cmd == "/exportar":
        if not s["log"]:
            await cl.Message(content="ℹ️ Nenhuma mensagem nesta sessão.").send()
            return

        linhas = [
            "# Log de Debug — Agente UEMA v3",
            f"# Gerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Modelo: {settings.GEMINI_MODEL if _MODULOS_OK else 'N/A'}",
            f"# Mensagens: {s['msgs']} · Tokens ~{s['tokens']}",
            "",
        ]
        for i, h in enumerate(s["log"], 1):
            linhas += [
                f"─── [{i}] {h['ts']} | {h['modo']} | {h['latencia']}ms",
                f">>> {h['pergunta']}",
                f"<<< {h['resposta']}",
                "",
            ]

        await cl.Message(
            content="📄 Log exportado:",
            elements=[cl.File(
                name=f"debug_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                content="\n".join(linhas).encode("utf-8"),
                mime="text/plain",
            )]
        ).send()

    else:
        await cl.Message(content=f"❓ Comando desconhecido: `{cmd}`. Usa `/ajuda`.").send()