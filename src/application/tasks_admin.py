"""
application/tasks_admin.py — Tasks Celery Admin (v2 — Validação + Auto-Config)
================================================================================

MUDANÇAS v2 vs v1:
───────────────────
  ADICIONADO:
    - Validação do documento ANTES de ingerir (document_validator.py)
    - Auto-config: parsing_instruction gerada automaticamente pelo nome do ficheiro
    - Feedback detalhado ao admin (tipo detectado, parser escolhido, chunks)
    - Mensagem de ajuda no !ingerir sem ficheiro (tipos aceites + formato esperado)
    - Protecção contra ficheiros corrompidos, muito grandes e scans

  MANTIDO:
    - Download via Evolution API /chat/getBase64FromMediaMessage
    - Celery task assíncrona
    - Confirmação ao admin via WhatsApp
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

_PASTA_UPLOADS = os.path.join(settings.DATA_DIR, "uploads")


# =============================================================================
# Task: Ingestão de documento via WhatsApp
# =============================================================================

@celery_app.task(name="ingerir_documento_whatsapp", bind=True, max_retries=2)
def ingerir_documento_task(self, identity: dict) -> None:
    """
    Baixa o ficheiro via Evolution API, valida, auto-configura e ingere.

    FLUXO v2:
      1. Download base64 via /chat/getBase64FromMediaMessage
      2. Salva em /dados/uploads/
      3. Valida com document_validator (formato, tamanho, estrutura)
      4. Se inválido → avisa admin com motivo específico
      5. Se válido   → usa config gerado automaticamente
      6. Ingere no Redis Stack
      7. Confirmação detalhada ao admin
    """
    chat_id  = identity.get("chat_id", "")
    key_id   = identity.get("msg_key_id", "")
    user_id  = identity.get("sender_phone", "admin")

    logger.info("📥 [ADMIN] Download doc | key=%s | user=%s", key_id[:20], user_id)

    try:
        # ── 1. Download ───────────────────────────────────────────────────────
        b64, mimetype, nome_original = _baixar_media_evolution(key_id)
        if not b64:
            _enviar(chat_id, "❌ Não consegui baixar o ficheiro. Verifica se o ficheiro ainda existe e tenta reenviar.")
            return

        # ── 2. Salva em disco ─────────────────────────────────────────────────
        from src.rag.document_validator import _detectar_extensao
        ext  = _detectar_extensao(nome_original, mimetype, "")
        if not ext:
            from src.rag.document_validator import tipos_aceites_mensagem
            _enviar(chat_id, f"❌ Formato não reconhecido: `{nome_original}`\n\n{tipos_aceites_mensagem()}")
            return

        hash_id  = hashlib.md5(b64[:500].encode()).hexdigest()[:8]
        nome_seg = _sanitizar_nome(nome_original or f"doc_{hash_id}{ext}")
        if not nome_seg.lower().endswith(ext):
            nome_seg = f"{os.path.splitext(nome_seg)[0]}{ext}"

        os.makedirs(_PASTA_UPLOADS, exist_ok=True)
        caminho = os.path.join(_PASTA_UPLOADS, nome_seg)

        conteudo_bytes = base64.b64decode(b64)
        with open(caminho, "wb") as f:
            f.write(conteudo_bytes)

        tamanho_kb = len(conteudo_bytes) // 1024
        logger.info("💾 Salvo: %s (%d KB)", caminho, tamanho_kb)

        # ── 3. Validação ──────────────────────────────────────────────────────
        from src.rag.document_validator import (
            validar_documento,
            formatar_resultado_para_whatsapp,
        )

        resultado = validar_documento(caminho, mimetype, nome_original)

        if not resultado.valido:
            _enviar(chat_id, resultado.motivo_rejeicao)
            # Remove ficheiro inválido
            try:
                os.remove(caminho)
            except Exception:
                pass
            return

        # Informa ao admin sobre o que foi detectado
        _enviar(chat_id, formatar_resultado_para_whatsapp(resultado, nome_seg))

        # ── 4. Regista no DOCUMENT_CONFIG com auto-config ─────────────────────
        from src.rag.ingestion import DOCUMENT_CONFIG
        config_auto = resultado.config_sugerido.copy()
        DOCUMENT_CONFIG[nome_seg] = config_auto

        logger.info(
            "📋 Auto-config: '%s' → doc_type=%s | parser=%s | chunk=%d",
            nome_seg, config_auto.get("doc_type"), config_auto.get("parser", "auto"),
            config_auto.get("chunk_size", 400),
        )

        # ── 5. Ingere ─────────────────────────────────────────────────────────
        t0      = time.monotonic()
        ingestor= _get_ingestor()
        n_chunks= ingestor._ingerir_ficheiro(caminho)
        ms      = int((time.monotonic() - t0) * 1000)

        # ── 6. Confirmação final ──────────────────────────────────────────────
        if n_chunks > 0:
            avisos = "\n".join(resultado.avisos) if resultado.avisos else ""
            _enviar(
                chat_id,
                f"✅ *Ingestão concluída!*\n\n"
                f"📄 Ficheiro: `{nome_seg}`\n"
                f"📦 Tamanho: {tamanho_kb} KB\n"
                f"🧩 Chunks gerados: *{n_chunks}*\n"
                f"⏱  Tempo: {ms}ms\n"
                f"📋 Tipo: {resultado.categoria} → `{resultado.doc_type}`\n"
                f"⚙️  Parser: `{resultado.parser_sugerido}`\n"
                + (f"\n{avisos}" if avisos else "") +
                f"\n\n🔍 Já disponível para busca! "
                f"Testa com uma pergunta relacionada.",
            )
        else:
            _enviar(
                chat_id,
                f"⚠️  *Ficheiro recebido mas 0 chunks gerados.*\n\n"
                f"Possíveis causas:\n"
                f"• PDF é scan/imagem sem OCR → tenta com `PDF_PARSER=llamaparse`\n"
                f"• CSV vazio ou com apenas 1 coluna\n"
                f"• Ficheiro corrompido\n\n"
                f"Ficheiro: `{nome_seg}` ({tamanho_kb} KB)",
            )

    except Exception as e:
        logger.exception("❌ Falha na ingestão via WhatsApp: %s", e)
        _enviar(chat_id, f"❌ Erro técnico ao ingerir: `{str(e)[:100]}`\nContacta o suporte.")


# =============================================================================
# Task: Comandos Admin
# =============================================================================

@celery_app.task(name="executar_comando_admin", bind=True)
def executar_comando_admin_task(self, chat_id: str, parametro: str, user_id: str) -> None:
    """Executa comandos admin assíncronos."""
    logger.info("⚙️  [ADMIN] Comando: %s | user=%s", parametro, user_id)
    try:
        if parametro == "LIMPAR_CACHE":
            _cmd_limpar_cache(chat_id)
        elif parametro == "TOOLS":
            _cmd_tools(chat_id)
        elif parametro.startswith("RAGAS:"):
            _cmd_exportar_ragas(chat_id, parametro.split(":", 1)[1] or None)
        elif parametro.startswith("FATOS:"):
            _cmd_fatos(chat_id, parametro.split(":", 1)[1] or user_id)
        elif parametro == "RELOAD":
            _cmd_reload(chat_id)
        elif parametro.startswith("INGERIR_FICHEIRO:"):
            nome = parametro.split(":", 1)[1]
            _cmd_ingerir_por_nome(chat_id, nome)
    except Exception as e:
        logger.exception("❌ Comando admin '%s': %s", parametro, e)
        _enviar(chat_id, f"❌ Erro: `{str(e)[:100]}`")


# =============================================================================
# Implementações dos comandos
# =============================================================================

def _cmd_limpar_cache(chat_id: str) -> None:
    from src.infrastructure.semantic_cache import invalidar_cache_rota
    from src.domain.entities import Rota
    total = sum(invalidar_cache_rota(r.value) for r in Rota)
    _enviar(chat_id, f"🗑️  *Cache limpo!*\n• Entradas removidas: {total}")


def _cmd_tools(chat_id: str) -> None:
    from src.domain.semantic_router import listar_tools_registadas
    tools = listar_tools_registadas()
    if not tools:
        _enviar(chat_id, "⚠️  Nenhuma tool registada.")
        return
    linhas = [f"🔧 *{len(tools)} Tools registadas:*\n"]
    for t in tools:
        linhas.append(f"• `{t['name']}`\n  {t['description'][:70]}...")
    _enviar(chat_id, "\n".join(linhas))


def _cmd_exportar_ragas(chat_id: str, target: str | None) -> None:
    r = get_redis_text()
    try:
        raw  = r.lrange("metrics:respostas", 0, 99)
        logs = [json.loads(l) for l in raw]
    except Exception as e:
        _enviar(chat_id, f"❌ Erro ao ler logs: {e}")
        return
    if target:
        logs = [l for l in logs if l.get("user_id") == target]
    if not logs:
        _enviar(chat_id, f"⚠️  Sem logs {'para ' + target if target else 'de produção'}.")
        return
    dataset = [{
        "question": l.get("pergunta", ""),
        "answer":   l.get("resposta", ""),
        "contexts": l.get("contextos", []),
        "ground_truth": "",
    } for l in logs]
    path = f"/app/dados/ragas_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        _enviar(chat_id, f"📊 *Dataset RAGAS exportado!*\n• Casos: {len(dataset)}\n• Ficheiro: `{path}`")
    except Exception as e:
        _enviar(chat_id, f"❌ Falha ao salvar: {e}")


def _cmd_fatos(chat_id: str, user_id: str) -> None:
    from src.memory.long_term_memory import listar_todos_fatos
    fatos = listar_todos_fatos(user_id)
    if not fatos:
        _enviar(chat_id, f"ℹ️  Sem fatos para `{user_id}`.")
        return
    linhas = [f"🧠 *Fatos de `{user_id}` ({len(fatos)}):*\n"]
    for f in fatos[:10]:
        linhas.append(f"• {f}")
    if len(fatos) > 10:
        linhas.append(f"_...e mais {len(fatos) - 10} fatos._")
    _enviar(chat_id, "\n".join(linhas))


def _cmd_reload(chat_id: str) -> None:
    from src.agent.core import agent_core
    from src.tools import get_tools_ativas
    try:
        tools = get_tools_ativas()
        agent_core.inicializar(tools)
        _enviar(chat_id, f"🔄 AgentCore reiniciado com {len(tools)} tools.")
    except Exception as e:
        _enviar(chat_id, f"❌ Reload falhou: `{str(e)[:100]}`")


def _cmd_ingerir_por_nome(chat_id: str, nome: str) -> None:
    """Ingere ficheiro existente em /dados/ pelo nome."""
    from src.rag.ingestion import DOCUMENT_CONFIG, Ingestor
    from src.rag.document_validator import validar_documento, formatar_resultado_para_whatsapp

    caminho = os.path.join(settings.DATA_DIR, nome)
    if not os.path.exists(caminho):
        _enviar(chat_id, f"❌ Ficheiro `{nome}` não encontrado em `/dados/`.")
        return

    resultado = validar_documento(caminho, "", nome)
    if not resultado.valido:
        _enviar(chat_id, resultado.motivo_rejeicao)
        return

    if nome not in DOCUMENT_CONFIG:
        DOCUMENT_CONFIG[nome] = resultado.config_sugerido
        logger.info("📋 Auto-config para '%s': %s", nome, resultado.config_sugerido)

    _enviar(chat_id, f"📂 A ingerir `{nome}`...\n{formatar_resultado_para_whatsapp(resultado, nome)}")

    ingestor = _get_ingestor()
    n_chunks = ingestor._ingerir_ficheiro(caminho)

    if n_chunks > 0:
        _enviar(chat_id, f"✅ `{nome}` ingerido! *{n_chunks} chunks* no Redis.")
    else:
        _enviar(chat_id, f"⚠️  `{nome}`: 0 chunks gerados. Verifica o formato.")


# =============================================================================
# Utilitários
# =============================================================================

def _baixar_media_evolution(msg_key_id: str) -> tuple[str, str, str]:
    """Download de media via Evolution API."""
    url = (
        f"{settings.EVOLUTION_BASE_URL.rstrip('/')}"
        f"/chat/getBase64FromMediaMessage/{settings.EVOLUTION_INSTANCE_NAME}"
    )
    body    = {"message": {"key": {"id": msg_key_id}}}
    headers = {"Content-Type": "application/json", "apikey": settings.EVOLUTION_API_KEY}

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        b64      = data.get("base64", "")
        mimetype = data.get("mimetype", "application/octet-stream")
        nome     = data.get("fileName", "documento")
        if not b64:
            logger.warning("⚠️  Evolution retornou base64 vazio para key=%s", msg_key_id[:20])
        return b64, mimetype, nome
    except Exception as e:
        logger.error("❌ Falha ao baixar media: %s", e)
        return "", "", ""


def _enviar(chat_id: str, texto: str) -> None:
    """Envia mensagem ao admin via Evolution API."""
    import asyncio
    try:
        from src.services.evolution_service import EvolutionService
        svc = EvolutionService()
        asyncio.run(svc.enviar_mensagem(chat_id, texto))
    except Exception as e:
        logger.warning("⚠️  Falha ao enviar confirmação: %s", e)


def _sanitizar_nome(nome: str) -> str:
    import re
    return re.sub(r"[^\w\-_. ]", "_", nome).strip()[:100]


def _get_ingestor():
    from src.rag.ingestion import Ingestor
    return Ingestor()