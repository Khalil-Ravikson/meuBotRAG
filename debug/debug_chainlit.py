"""
================================================================================
debug/debug_chainlit.py — Painel de Debug do Agente UEMA
================================================================================

Painel interativo para testar o agente SEM WhatsApp.
Usa os mesmos módulos de produção: agent_core, redis_memory, ingestor.

USO — rode SEMPRE da raiz do projeto:
    cd /home/arch/Projects/meuBotRag/meuBotRAG
    chainlit run debug/debug_chainlit.py --port 8001

COMANDOS NO CHAT:
    /ajuda          lista todos os comandos
    /status         configuração atual + LangSmith + HF_TOKEN
    /limpar         limpa histórico Redis + estado do menu do usuário de teste
    /diagnostico    verifica sources no banco vetorial (debug do "Não encontrei")
    /modo agente    fluxo completo: menu → router → agent_core
    /modo direto    só o agent_core, sem menu/router
    /ingerir        força re-ingestão dos PDFs

SOBRE O chainlit.toml:
    Fica em debug/chainlit.toml e configura visual (nome, cores, avatar).
    NÃO vai para o Docker — é só para este painel de dev.

SOBRE O LANGSMITH:
    Se LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true estiverem no .env,
    cada mensagem aqui aparece rastreada em https://smith.langchain.com
================================================================================
"""
from __future__ import annotations

import sys
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# ── Verificação de versão ─────────────────────────────────────────────────────
if sys.version_info >= (3, 13):
    print("❌  Python 3.13+ não suportado pelo Chainlit.")
    print("    Use Python 3.11 ou 3.12: pyenv install 3.11.9")
    sys.exit(1)

# ── Resolve raiz do projeto: funciona de debug/ ou da raiz ───────────────────
_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parent if _AQUI.name == "debug" else _AQUI
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import chainlit as cl

# ── Imports dos módulos de produção ───────────────────────────────────────────
_MODULOS_OK   = True
_ERRO_IMPORT  = None
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
        get_contexto, set_contexto, limpar_historico, clear_tudo,
    )
    from src.rag.ingestor                import Ingestor, PDF_CONFIG
    from src.rag.vector_store            import diagnosticar as vs_diagnosticar
    from src.tools                       import get_tools_ativas
except ImportError as e:
    _MODULOS_OK  = False
    _ERRO_IMPORT = str(e)

# ── Contador de tokens (aproximado) ───────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _tokens(t: str) -> int:
        return len(_enc.encode(str(t)))
except Exception:
    def _tokens(t: str) -> int:
        return len(str(t)) // 4

logging.basicConfig(level=logging.WARNING)

# ID fixo para o usuário de teste — isola histórico de produção
_DEBUG_USER = "debug_chainlit"


# =============================================================================
# Estado da sessão Chainlit
# =============================================================================

def _novo_estado() -> dict:
    return {
        "modo":           "agente",   # "agente" | "direto"
        "iniciado_em":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "msgs":           0,
        "tokens":         0,
        "latencia_total": 0,
        "log":            [],          # para /exportar
    }


# =============================================================================
# on_chat_start — executa quando o usuário abre o painel
# =============================================================================

@cl.on_chat_start
async def on_start():
    cl.user_session.set("s", _novo_estado())

    if not _MODULOS_OK:
        await cl.Message(content=(
            f"⚠️ **Erro ao importar módulos**\n\n```\n{_ERRO_IMPORT}\n```\n\n"
            "Execute da raiz do projeto:\n"
            "```bash\ncd /caminho/para/meuBotRAG\n"
            "chainlit run debug/debug_chainlit.py --port 8001\n```"
        )).send()
        return

    # Inicializa o agente se ainda não foi feito
    if not agent_core._agent_with_history:
        async with cl.Step(name="🔧 Inicializando agente") as step:
            try:
                ingestor = Ingestor()
                await asyncio.to_thread(ingestor.ingerir_se_necessario)
                tools = get_tools_ativas()
                await asyncio.to_thread(agent_core.inicializar, tools)
                step.output = f"✅ {len(tools)} tools carregadas."
            except Exception as e:
                step.output = f"❌ {e}"

    ls = (
        f"✅ [ver traces](https://smith.langchain.com) → `{settings.LANGCHAIN_PROJECT}`"
        if settings.langsmith_ativo else "❌ desativado (adicione LANGCHAIN_API_KEY no .env)"
    )

    await cl.Message(content=(
        "## 🎓 Debug — Agente UEMA\n\n"
        f"**Modo:** `agente` &nbsp;|&nbsp; **Modelo:** `{settings.GROQ_MODEL}`\n\n"
        f"**LangSmith:** {ls}\n"
        f"**HF_TOKEN:** {'✅ configurado' if settings.HF_TOKEN else '⚠️ ausente (download anônimo)'}\n\n"
        "---\nDigite uma mensagem ou use um comando:\n"
        "`/ajuda` · `/status` · `/limpar` · `/diagnostico` · `/modo direto` · `/modo agente` · `/ingerir`"
    )).send()


# =============================================================================
# on_message — processa cada mensagem do usuário
# =============================================================================

@cl.on_message
async def on_message(message: cl.Message):
    texto = message.content.strip()
    s     = cl.user_session.get("s") or _novo_estado()

    if texto.startswith("/"):
        await _cmd(texto, s)
    elif not _MODULOS_OK:
        await cl.Message(content="⚠️ Módulos não carregados.").send()
    elif s["modo"] == "agente":
        await _modo_agente(texto, s)
    else:
        await _modo_direto(texto, s)

    cl.user_session.set("s", s)


# =============================================================================
# Modos de resposta
# =============================================================================

async def _modo_agente(texto: str, s: dict) -> None:
    """Fluxo completo: menu → router → agent_core."""
    t0        = time.time()
    modo_menu = get_estado_menu(_DEBUG_USER)

    # domain/menu.py — stateless
    resultado = processar_mensagem(texto, modo_menu)

    # Resposta direta do menu (sem LLM)
    if resultado["type"] in ("menu_principal", "submenu"):
        set_estado_menu(_DEBUG_USER, resultado["novo_estado"])
        await cl.Message(content=resultado["content"], author="📋 Menu").send()
        _log(s, texto, resultado["content"], 0, "menu")
        return

    # Atualiza estado do menu
    novo = resultado["novo_estado"]
    (clear_estado_menu if novo == EstadoMenu.MAIN else set_estado_menu)(_DEBUG_USER, *([novo] if novo != EstadoMenu.MAIN else []))

    # Router
    prompt_base  = resultado["prompt"] or texto
    rota         = analisar(prompt_base, modo_menu)
    ctx          = get_contexto(_DEBUG_USER)
    prompt_final = montar_prompt_enriquecido(prompt_base, rota, ctx)

    # Mostra rota como nota de debug
    await cl.Message(
        content=f"`🔍 Rota: {rota.value}` · `Menu: {modo_menu.value}`",
        author="Router"
    ).send()

    # AgentState
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
        step.output = f"Latência: **{latencia}ms** · Tokens ~{toks} · {'✅' if resp.sucesso else '❌'}"

    set_contexto(_DEBUG_USER, {"ultima_intencao": rota.value})
    await cl.Message(content=resp.conteudo).send()
    _log(s, texto, resp.conteudo, latencia, f"agente/{rota.value}")


async def _modo_direto(texto: str, s: dict) -> None:
    """Direto ao agent_core, sem menu nem router."""
    t0    = time.time()
    state = AgentState(
        user_id=_DEBUG_USER, session_id=_DEBUG_USER,
        mensagem_original=texto, chat_id="debug",
    )
    async with cl.Step(name="🤖 Agent [direto]") as step:
        resp     = await asyncio.to_thread(agent_core.responder, state)
        latencia = int((time.time() - t0) * 1000)
        step.output = f"Latência: **{latencia}ms** · {'✅' if resp.sucesso else '❌'}"

    await cl.Message(content=resp.conteudo).send()
    _log(s, texto, resp.conteudo, latencia, "direto")


def _log(s: dict, pergunta: str, resposta: str, lat: int, modo: str):
    s["msgs"]           += 1
    s["tokens"]         += _tokens(pergunta) + _tokens(resposta)
    s["latencia_total"] += lat
    s["log"].append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "modo": modo, "latencia": lat,
        "pergunta": pergunta, "resposta": resposta,
    })


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
            "| `/status` | Config atual, LangSmith, HF_TOKEN |\n"
            "| `/limpar` | Limpa histórico Redis + estado do menu |\n"
            "| `/diagnostico` | Sources no banco (debug do 'Não encontrei') |\n"
            "| `/modo agente` | Fluxo completo: menu → router → agent |\n"
            "| `/modo direto` | Só agent_core, sem menu/router |\n"
            "| `/ingerir` | Força re-ingestão dos PDFs |\n"
            "| `/exportar` | Baixa log da sessão em .txt |\n"
        )).send()

    elif cmd == "/status":
        msgs = s["msgs"]
        lat  = (s["latencia_total"] // msgs) if msgs else 0
        ls   = (
            f"✅ projeto `{settings.LANGCHAIN_PROJECT}` · [abrir dashboard](https://smith.langchain.com)"
            if settings.langsmith_ativo else "❌ desativado"
        )
        await cl.Message(content=(
            f"## Status\n\n"
            f"**Modo:** `{s['modo']}`\n"
            f"**Modelo:** `{settings.GROQ_MODEL}`\n"
            f"**Agente pronto:** {'✅' if agent_core._agent_with_history else '❌'}\n"
            f"**LangSmith:** {ls}\n"
            f"**HF_TOKEN:** {'✅ configurado' if settings.HF_TOKEN else '⚠️ ausente'}\n\n"
            f"**Sessão:**\n"
            f"- Mensagens: {msgs} · Tokens ~{s['tokens']} · Lat. média: {lat}ms\n"
            f"- Iniciado: {s['iniciado_em']}\n"
        )).send()

    elif cmd == "/limpar":
        if _MODULOS_OK:
            clear_tudo(_DEBUG_USER)
            await cl.Message(content="🗑️ Histórico + estado do menu limpos.").send()
        else:
            await cl.Message(content="⚠️ Módulos não carregados.").send()

    elif cmd == "/diagnostico":
        if not _MODULOS_OK:
            await cl.Message(content="⚠️ Módulos não carregados.").send()
            return
        async with cl.Step(name="🔍 Verificando banco vetorial") as step:
            sources = await asyncio.to_thread(vs_diagnosticar)
            step.output = str(sources)
        esperados = set(PDF_CONFIG.keys())
        faltam    = esperados - sources
        linhas    = ["### Sources no banco vetorial\n"]
        for src in sorted(sources):
            icone = "✅" if src in esperados else "⚠️ não esperado"
            linhas.append(f"- `{src}` {icone}")
        if faltam:
            linhas.append(f"\n❌ **Não encontrados no banco:** {', '.join(f'`{f}`' for f in faltam)}")
            linhas.append("\n💡 Rode `/ingerir` para processar os PDFs que faltam.")
        else:
            linhas.append("\n✅ Todos os arquivos esperados estão no banco.")
        await cl.Message(content="\n".join(linhas)).send()

    elif cmd == "/modo":
        novo = partes[1] if len(partes) > 1 else ""
        if novo in ("agente", "direto"):
            s["modo"] = novo
            await cl.Message(content=f"✅ Modo alterado para `{novo}`.").send()
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
            linhas += [
                f"─── [{i}] {h['ts']} | {h['modo']} | {h['latencia']}ms",
                f">>> {h['pergunta']}",
                f"<<< {h['resposta']}", "",
            ]
        await cl.Message(
            content="📄 Log da sessão:",
            elements=[cl.File(
                name=f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                content="\n".join(linhas).encode("utf-8"),
                mime="text/plain",
            )]
        ).send()

    else:
        await cl.Message(content=f"❓ Comando desconhecido: `{cmd}`. Use `/ajuda`.").send()