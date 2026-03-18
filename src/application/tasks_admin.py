"""
application/tasks_admin.py — Tasks Celery para Comandos Admin (v1.0)
======================================================================

TASKS DISPONÍVEIS:
───────────────────
  ingerir_documento_task   → baixa media da Evolution API + ingere no Redis Stack
  executar_comando_admin   → processa comandos !status, !limpar_cache, !tools, etc.
  exportar_logs_ragas      → converte logs de produção para dataset RAGAS

FLUXO DE INGESTÃO VIA WHATSAPP:
─────────────────────────────────
  1. Admin envia PDF/CSV/DOCX com legenda "!ingerir"
  2. DevGuard extrai msg_key_id do payload (key.id da mensagem)
  3. SecurityGuard detecta ação INGERIR_DOC
  4. handle_message.py dispara ingerir_documento_task.delay(identity)
  5. Celery worker:
       a) Chama Evolution API: POST /chat/getBase64FromMediaMessage
          com {"message": {"key": {"id": msg_key_id}}}
       b) Decodifica base64 → bytes do ficheiro
       c) Detecta extensão pelo mimetype
       d) Salva em /app/dados/uploads/{hash}_{nome_original}
       e) Adiciona ao DOCUMENT_CONFIG dinamicamente
       f) Chama Ingestor().ingerir_ficheiro()
       g) Envia confirmação ao admin via Evolution API

SUPORTE DE FORMATOS (via Evolution API base64):
────────────────────────────────────────────────
  PDF  → .pdf  (documentMessage com mimetype application/pdf)
  CSV  → .csv  (documentMessage com mimetype text/csv)
  DOCX → .docx (documentMessage com mimetype application/vnd.openxmlformats...)
  XLSX → .xlsx (documentMessage com mimetype application/vnd.openxmlformats...)
  TXT  → .txt  (documentMessage com mimetype text/plain)

FLUXO LOGS → RAGAS:
────────────────────
  Os logs de produção (metrics:respostas) são automaticamente convertidos
  em casos de teste no formato RAGAS:
    {"question": "...", "answer": "...", "contexts": [...], "ground_truth": ""}
  Salvos em /app/dados/ragas_dataset_{data}.json para alimentar rag_eval.py.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from datetime import datetime

import httpx

from src.application.tasks import celery_app
from src.infrastructure.redis_client import get_redis_text
from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

# Tipos de ficheiro suportados (mimetype → extensão)
_MIME_TO_EXT: dict[str, str] = {
    "application/pdf":                                               ".pdf",
    "text/csv":                                                      ".csv",
    "text/plain":                                                    ".txt",
    "application/msword":                                            ".docx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel":                                      ".xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/octet-stream":                                      ".pdf",  # fallback
}

_PASTA_UPLOADS = os.path.join(settings.DATA_DIR, "uploads")


# ─────────────────────────────────────────────────────────────────────────────
# Task: Ingestão de documento via WhatsApp
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(name="ingerir_documento_whatsapp", bind=True, max_retries=2)
def ingerir_documento_task(self, identity: dict) -> None:
    """
    Baixa o ficheiro via Evolution API, salva em /dados/uploads/ e ingere.

    identity deve conter:
      msg_key_id:  ID da mensagem para download
      chat_id:     destino para confirmação
      sender_phone: para logging
      body:        legenda/caption da mensagem (para nome do doc)
    """
    chat_id    = identity.get("chat_id", "")
    key_id     = identity.get("msg_key_id", "")
    user_id    = identity.get("sender_phone", "admin")

    logger.info("📥 [ADMIN] Iniciando download doc | key_id=%s | user=%s", key_id[:20], user_id)

    try:
        # ── 1. Download base64 via Evolution API ─────────────────────────────
        base64_data, mimetype, nome_original = _baixar_media_evolution(key_id)

        if not base64_data:
            _enviar_resposta(chat_id, "❌ Não consegui baixar o ficheiro. Tente reenviar.")
            return

        # ── 2. Detecta extensão pelo mimetype ────────────────────────────────
        ext = _MIME_TO_EXT.get(mimetype, "")
        if not ext:
            ext = _ext_do_nome(nome_original) or ".pdf"

        # ── 3. Gera nome único e salva ───────────────────────────────────────
        hash_id  = hashlib.md5(base64_data[:1000].encode()).hexdigest()[:8]
        nome_seg = _sanitizar_nome(nome_original or f"doc_whatsapp_{hash_id}")
        if not nome_seg.endswith(ext):
            nome_seg = f"{nome_seg}{ext}"

        os.makedirs(_PASTA_UPLOADS, exist_ok=True)
        caminho = os.path.join(_PASTA_UPLOADS, nome_seg)

        with open(caminho, "wb") as f:
            f.write(base64.b64decode(base64_data))

        tamanho_kb = os.path.getsize(caminho) // 1024
        logger.info("💾 Ficheiro salvo: %s (%d KB)", caminho, tamanho_kb)

        # ── 4. Adiciona ao DOCUMENT_CONFIG e ingere ──────────────────────────
        nome_chunks = _ingerir_ficheiro_admin(caminho, nome_seg, ext)

        # ── 5. Confirmação ao admin ───────────────────────────────────────────
        if nome_chunks > 0:
            _enviar_resposta(
                chat_id,
                f"✅ *Documento ingerido com sucesso!*\n\n"
                f"📄 Ficheiro: `{nome_seg}`\n"
                f"📦 Tamanho: {tamanho_kb} KB\n"
                f"🧩 Chunks gerados: {nome_chunks}\n"
                f"🔍 Já disponível para busca híbrida!\n\n"
                f"Use `/banco/sources` para confirmar.",
            )
        else:
            _enviar_resposta(
                chat_id,
                f"⚠️  Ficheiro recebido mas 0 chunks gerados.\n"
                f"Verifica se o formato é suportado (PDF, CSV, DOCX, XLSX, TXT).\n"
                f"Ficheiro: `{nome_seg}`",
            )

    except Exception as e:
        logger.exception("❌ Falha na ingestão via WhatsApp: %s", e)
        _enviar_resposta(chat_id, f"❌ Erro ao ingerir documento: {str(e)[:100]}")


# ─────────────────────────────────────────────────────────────────────────────
# Task: Comandos Admin
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(name="executar_comando_admin", bind=True)
def executar_comando_admin_task(self, chat_id: str, parametro: str, user_id: str) -> None:
    """
    Executa comandos administrativos disparados via WhatsApp.

    parametros suportados:
      LIMPAR_CACHE      → invalida Semantic Cache
      STATUS            → retorna estado do sistema (síncrono no handle_message)
      TOOLS             → lista tools registadas
      RAGAS:{user_id}   → exporta logs para dataset RAGAS
      FATOS:{user_id}   → lista fatos de um utilizador
      RELOAD            → reinicia AgentCore
    """
    logger.info("⚙️  [ADMIN] Comando: %s | user=%s", parametro, user_id)

    try:
        if parametro == "LIMPAR_CACHE":
            _cmd_limpar_cache(chat_id)

        elif parametro == "TOOLS":
            _cmd_tools(chat_id)

        elif parametro.startswith("RAGAS:"):
            target = parametro.split(":", 1)[1]
            _cmd_exportar_ragas(chat_id, target or None)

        elif parametro.startswith("FATOS:"):
            target = parametro.split(":", 1)[1]
            _cmd_fatos(chat_id, target or user_id)

        elif parametro == "RELOAD":
            _cmd_reload(chat_id)

    except Exception as e:
        logger.exception("❌ Erro no comando admin '%s': %s", parametro, e)
        _enviar_resposta(chat_id, f"❌ Erro ao executar comando: {str(e)[:100]}")


# ─────────────────────────────────────────────────────────────────────────────
# Implementações dos comandos
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_limpar_cache(chat_id: str) -> None:
    from src.infrastructure.semantic_cache import cache_stats, invalidar_cache_rota
    from src.domain.entities import Rota

    total = 0
    for rota in Rota:
        total += invalidar_cache_rota(rota.value)

    stats = cache_stats()
    _enviar_resposta(
        chat_id,
        f"🗑️  *Semantic Cache limpo!*\n"
        f"• Entradas removidas: {total}\n"
        f"• Índice: `{stats.get('index_name', '?')}`\n"
        f"• Dimensão vector: {stats.get('vector_dim', '?')}",
    )


def _cmd_tools(chat_id: str) -> None:
    from src.domain.semantic_router import listar_tools_registadas
    tools = listar_tools_registadas()
    if not tools:
        _enviar_resposta(chat_id, "⚠️  Nenhuma tool registada no Redis.")
        return
    linhas = [f"🔧 *Tools registadas ({len(tools)}):*\n"]
    for t in tools:
        linhas.append(f"• `{t['name']}`\n  {t['description'][:80]}...")
    _enviar_resposta(chat_id, "\n".join(linhas))


def _cmd_exportar_ragas(chat_id: str, target_user: str | None) -> None:
    """
    Converte logs de produção (metrics:respostas) em dataset RAGAS.

    Formato de saída:
      [{
        "question": "quando é a matrícula?",
        "answer": "A matrícula de veteranos ocorre de...",
        "contexts": ["[CALENDÁRIO ACADÊMICO UEMA 2026] ..."],
        "ground_truth": ""  ← preencher manualmente para eval real
      }]
    """
    r = get_redis_text()
    try:
        logs_raw = r.lrange("metrics:respostas", 0, 99)
        logs = [json.loads(l) for l in logs_raw]
    except Exception as e:
        _enviar_resposta(chat_id, f"❌ Erro ao ler logs: {e}")
        return

    # Filtra por user se especificado
    if target_user:
        logs = [l for l in logs if l.get("user_id") == target_user]

    if not logs:
        _enviar_resposta(chat_id, f"⚠️  Sem logs {'para ' + target_user if target_user else 'de produção'}.")
        return

    # Monta dataset RAGAS
    dataset = []
    for log in logs:
        entry = {
            "question":    log.get("pergunta", ""),
            "answer":      log.get("resposta",  ""),
            "contexts":    log.get("contextos", []),
            "ground_truth": "",  # para preenchimento manual
            "user_id":     log.get("user_id", ""),
            "rota":        log.get("rota", ""),
            "latencia_ms": log.get("latencia_ms", 0),
            "tokens_total": log.get("tokens_total", 0),
        }
        dataset.append(entry)

    # Salva em disco
    data_str = datetime.now().strftime("%Y%m%d_%H%M")
    path     = f"/app/dados/ragas_dataset_{data_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    _enviar_resposta(
        chat_id,
        f"📊 *Dataset RAGAS exportado!*\n"
        f"• Casos: {len(dataset)}\n"
        f"• Ficheiro: `{path}`\n"
        f"• Próximo passo: preenche `ground_truth` e corre `rag_eval.py`\n\n"
        f"_Use `/banco/sources` para confirmar que o dataset foi gerado._",
    )


def _cmd_fatos(chat_id: str, user_id: str) -> None:
    from src.memory.long_term_memory import listar_todos_fatos
    fatos = listar_todos_fatos(user_id)
    if not fatos:
        _enviar_resposta(chat_id, f"ℹ️  Sem fatos para `{user_id}`.")
        return
    linhas = [f"🧠 *Fatos de `{user_id}` ({len(fatos)}):*\n"]
    for f in fatos[:10]:
        linhas.append(f"• {f}")
    if len(fatos) > 10:
        linhas.append(f"_...e mais {len(fatos) - 10} fatos._")
    _enviar_resposta(chat_id, "\n".join(linhas))


def _cmd_reload(chat_id: str) -> None:
    from src.agent.core import agent_core
    from src.tools import get_tools_ativas
    try:
        tools = get_tools_ativas()
        agent_core.inicializar(tools)
        _enviar_resposta(chat_id, f"🔄 AgentCore reiniciado com {len(tools)} tools.")
    except Exception as e:
        _enviar_resposta(chat_id, f"❌ Reload falhou: {str(e)[:100]}")


# ─────────────────────────────────────────────────────────────────────────────
# Ingestão interna
# ─────────────────────────────────────────────────────────────────────────────

def _ingerir_ficheiro_admin(caminho: str, nome: str, ext: str) -> int:
    """
    Adiciona o ficheiro ao DOCUMENT_CONFIG dinamicamente e ingere.
    Usa config genérico baseado na extensão.
    """
    from src.rag.ingestion import DOCUMENT_CONFIG, Ingestor

    # Config automático por tipo de ficheiro
    tipo_map = {
        ".pdf":  ("geral", 400, 60),
        ".csv":  ("geral", 300, 40),
        ".docx": ("geral", 400, 60),
        ".xlsx": ("geral", 300, 40),
        ".txt":  ("geral", 350, 50),
    }
    doc_type, chunk_size, overlap = tipo_map.get(ext, ("geral", 400, 60))
    label = nome.replace(ext, "").replace("_", " ").upper()

    # Regista dinamicamente no DOCUMENT_CONFIG
    DOCUMENT_CONFIG[nome] = {
        "doc_type":   doc_type,
        "titulo":     label,
        "chunk_size": chunk_size,
        "overlap":    overlap,
        "label":      label,
    }
    logger.info("📋 Config dinâmico registado: '%s'", nome)

    # Usa a pasta de uploads como data_dir temporário
    settings_orig = settings.DATA_DIR
    try:
        ingestor = Ingestor()
        return ingestor._ingerir_ficheiro(caminho)
    except Exception as e:
        logger.exception("❌ _ingerir_ficheiro_admin: %s", e)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Evolution API — Download de media
# ─────────────────────────────────────────────────────────────────────────────

def _baixar_media_evolution(msg_key_id: str) -> tuple[str, str, str]:
    """
    Chama POST /chat/getBase64FromMediaMessage/{instance} para obter o ficheiro.

    Retorna: (base64_data, mimetype, nome_original)
    Retorna ("", "", "") em caso de erro.

    Endpoint descoberto no JSON Postman:
      POST {{baseUrl}}/chat/getBase64FromMediaMessage/{{instance}}
      Body: {"message": {"key": {"id": "MSG_ID"}}}
    """
    url = (
        f"{settings.EVOLUTION_BASE_URL.rstrip('/')}"
        f"/chat/getBase64FromMediaMessage/{settings.EVOLUTION_INSTANCE_NAME}"
    )
    headers = {
        "Content-Type": "application/json",
        "apikey":       settings.EVOLUTION_API_KEY,
    }
    body = {"message": {"key": {"id": msg_key_id}}}

    logger.debug("📡 Evolution download | url=%s | key_id=%s", url, msg_key_id[:20])

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # A Evolution API retorna: {"base64": "...", "mimetype": "...", "fileName": "..."}
        b64      = data.get("base64", "")
        mimetype = data.get("mimetype", "application/octet-stream")
        nome     = data.get("fileName", "documento")

        if not b64:
            logger.warning("⚠️  Evolution retornou base64 vazio para key_id=%s", msg_key_id[:20])
            return "", "", ""

        logger.info("✅ Media baixada: %s | %s | %d chars base64", nome, mimetype, len(b64))
        return b64, mimetype, nome

    except httpx.HTTPStatusError as e:
        logger.error("❌ Evolution API HTTP %d para getBase64: %s", e.response.status_code, e.response.text[:200])
        return "", "", ""
    except Exception as e:
        logger.exception("❌ Falha ao baixar media: %s", e)
        return "", "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def _enviar_resposta(chat_id: str, texto: str) -> None:
    """Envia mensagem de confirmação ao admin via Evolution API."""
    import asyncio
    try:
        from src.services.evolution_service import EvolutionService
        svc = EvolutionService()
        asyncio.run(svc.enviar_mensagem(chat_id, texto))
    except Exception as e:
        logger.warning("⚠️  Falha ao enviar confirmação admin: %s", e)


def _sanitizar_nome(nome: str) -> str:
    """Remove caracteres inseguros do nome do ficheiro."""
    import re
    safe = re.sub(r"[^\w\-_. ]", "_", nome)
    return safe.strip()[:100]


def _ext_do_nome(nome: str) -> str:
    """Extrai extensão de um nome de ficheiro."""
    import os
    return os.path.splitext(nome)[1].lower() if nome else ""