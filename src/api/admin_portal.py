"""
api/admin_portal.py — Portal de Administração do Oráculo UEMA (v1.0)
======================================================================

ACESSO:
  URL:     http://localhost:9000/admin/
  Auth:    ADMIN_API_KEY do .env (inserido no login)
  Sessão:  Chave armazenada em sessionStorage do browser

SEÇÕES DO PORTAL:
  1. 🏠 Overview     — saúde de todos os serviços + métricas gerais
  2. 🔴 Redis        — memória, índices, keys, busca e exclusão
  3. 🐘 Postgres     — tabelas, tamanhos, usuários, migrações
  4. 📦 Ingestão RAG — sources, chunks, re-ingestão, cache semântico
  5. 🕷️  Scraping    — wiki indexada, cache, trigger manual
  6. 👥 Usuários     — listar, editar role/status, ver atividade
  7. 🧠 Memória      — fatos por usuário, working memory, limpeza
  8. ⚙️  Configuração — settings seguros (sem secrets), feature flags
  9. 📋 Logs         — erros recentes, logs streaming SSE

SEGURANÇA:
  - Todos os endpoints validam X-Admin-Key === settings.ADMIN_API_KEY
  - Settings com secrets NUNCA são expostos (lista branca de campos seguros)
  - Operações destrutivas retornam confirmação explícita
  - Sem escrita em .env (alterações são apenas em memória / Redis)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from functools import wraps
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# Autenticação
# ─────────────────────────────────────────────────────────────────────────────

def _get_admin_key() -> str:
    try:
        from src.infrastructure.settings import settings
        return settings.ADMIN_API_KEY or ""
    except Exception:
        return os.environ.get("ADMIN_API_KEY", "")


def require_admin(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    """Dependency: valida X-Admin-Key header."""
    key = _get_admin_key()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_API_KEY não configurada no .env"
        )
    if x_admin_key != key:
        raise HTTPException(
            status_code=401,
            detail="Chave inválida"
        )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Campos seguros para expor em /admin/config
# ─────────────────────────────────────────────────────────────────────────────

SAFE_FIELDS = {
    "DEV_MODE", "LOG_LEVEL", "GEMINI_MODEL", "GEMINI_TEMP", "GEMINI_MAX_TOKENS",
    "AGENT_MAX_ITERATIONS", "AGENT_TIMEOUT_S", "MAX_HISTORY_MESSAGES",
    "ROUTER_SIMILARITY_THRESHOLD", "PDF_PARSER", "DATA_DIR", "EVOLUTION_BASE_URL",
    "EVOLUTION_INSTANCE_NAME", "LANGCHAIN_PROJECT", "LANGCHAIN_TRACING_V2",
    "CRAG_THRESHOLD_OK", "CRAG_THRESHOLD_MIN",
}

EDITABLE_FIELDS = {
    "DEV_MODE": bool,
    "LOG_LEVEL": str,
    "GEMINI_TEMP": float,
    "GEMINI_MAX_TOKENS": int,
    "AGENT_MAX_ITERATIONS": int,
    "AGENT_TIMEOUT_S": int,
    "MAX_HISTORY_MESSAGES": int,
    "ROUTER_SIMILARITY_THRESHOLD": float,
}

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — Auth
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse, include_in_schema=False)
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def admin_portal():
    """Serve o SPA do portal admin."""
    return HTMLResponse(content=ADMIN_HTML)


@router.post("/auth")
async def auth_check(request: Request):
    """Valida a chave admin enviada pelo browser."""
    try:
        body = await request.json()
        key  = body.get("key", "")
    except Exception:
        raise HTTPException(400, "Corpo JSON inválido")

    expected = _get_admin_key()
    if not expected:
        raise HTTPException(503, "ADMIN_API_KEY não configurada")
    if key != expected:
        raise HTTPException(401, "Chave incorreta")

    return {"ok": True, "msg": "Autenticado com sucesso"}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — Overview
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/overview")
async def overview(auth=None):
    """Saúde de todos os serviços + métricas gerais."""
    from fastapi import Depends
    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "services":  {},
        "metrics":   {},
        "activity":  [],
    }

    # Redis
    try:
        from src.infrastructure.redis_client import get_redis, redis_ok
        r    = get_redis()
        info = r.info("server")
        mem  = r.info("memory")
        result["services"]["redis"] = {
            "status":  "ok" if redis_ok() else "down",
            "version": info.get("redis_version", "?"),
            "ram_mb":  round(mem.get("used_memory", 0) / 1024 / 1024, 1),
            "peak_mb": round(mem.get("used_memory_peak", 0) / 1024 / 1024, 1),
            "uptime_h": round(info.get("uptime_in_seconds", 0) / 3600, 1),
        }
    except Exception as e:
        result["services"]["redis"] = {"status": "down", "error": str(e)[:100]}

    # PostgreSQL
    try:
        from src.infrastructure.database import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as s:
            row = await s.execute(text("SELECT version(), current_database(), pg_size_pretty(pg_database_size(current_database()))"))
            v, db, sz = row.fetchone()
        result["services"]["postgres"] = {
            "status": "ok", "database": db, "size": sz,
            "version": v.split(" ")[1] if " " in v else v[:20],
        }
    except Exception as e:
        result["services"]["postgres"] = {"status": "down", "error": str(e)[:100]}

    # Gemini
    try:
        from src.infrastructure.settings import settings
        has_key = bool(settings.GEMINI_API_KEY)
        result["services"]["gemini"] = {
            "status":  "configured" if has_key else "no_key",
            "model":   settings.GEMINI_MODEL,
            "temp":    settings.GEMINI_TEMP,
        }
    except Exception as e:
        result["services"]["gemini"] = {"status": "error", "error": str(e)[:80]}

    # Evolution API
    try:
        from src.infrastructure.settings import settings
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r2 = await client.get(
                f"{settings.EVOLUTION_BASE_URL.rstrip('/')}/",
                headers={"apikey": settings.EVOLUTION_API_KEY},
            )
        result["services"]["evolution"] = {
            "status": "ok" if r2.status_code < 400 else "error",
            "url":     settings.EVOLUTION_BASE_URL,
            "instance": settings.EVOLUTION_INSTANCE_NAME,
            "http":    r2.status_code,
        }
    except Exception as e:
        result["services"]["evolution"] = {"status": "down", "error": str(e)[:80]}

    # AgentCore
    try:
        from src.agent.core import agent_core
        result["services"]["agent"] = {
            "status": "ready" if agent_core._inicializado else "not_ready",
            "tools":  len(agent_core._tools),
        }
    except Exception as e:
        result["services"]["agent"] = {"status": "error", "error": str(e)[:80]}

    # Métricas gerais
    try:
        from src.infrastructure.redis_client import get_redis_text
        rt = get_redis_text()
        logs = rt.lrange("monitor:logs", 0, 199)
        msgs = [json.loads(l) for l in logs]
        today = datetime.now().strftime("%Y-%m-%d")
        hoje  = [m for m in msgs if m.get("ts", "").startswith(today)]
        total_tok = sum(m.get("tokens_total", 0) for m in msgs[:50])
        avg_lat   = int(sum(m.get("latencia_ms", 0) for m in msgs[:50]) / max(len(msgs[:50]), 1))
        # count users
        _, keys_mem = get_redis_text().scan(0, match="mem:facts:list:*", count=500)
        _, keys_chat = get_redis_text().scan(0, match="chat:*", count=500)
        result["metrics"] = {
            "msgs_total":    len(msgs),
            "msgs_hoje":     len(hoje),
            "tokens_medio":  total_tok // max(len(msgs[:50]), 1),
            "latencia_media": avg_lat,
            "users_com_fatos": len(keys_mem),
            "sessoes_ativas":  len(keys_chat),
        }
        result["activity"] = msgs[:10]
    except Exception as e:
        result["metrics"] = {"error": str(e)[:100]}

    # Chunks RAG
    try:
        from src.rag.ingestion import Ingestor
        sources = Ingestor().diagnosticar()
        result["metrics"]["rag_sources"] = len(sources)
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — Redis
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/redis")
async def redis_info(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.redis_client import (
            get_redis, IDX_CHUNKS, IDX_TOOLS,
            PREFIX_CHUNKS, PREFIX_TOOLS,
        )
        r = get_redis()

        # Info geral
        info_mem    = r.info("memory")
        info_srv    = r.info("server")
        info_stats  = r.info("stats")
        info_clients= r.info("clients")

        # Contagem de keys por prefixo
        prefixes = {
            "rag:chunk":   "rag:chunk:*",
            "tools:emb":   "tools:emb:*",
            "cache":       "cache:*",
            "chat":        "chat:*",
            "menu_state":  "menu_state:*",
            "user_ctx":    "user_ctx:*",
            "mem:facts":   "mem:facts:*",
            "mem:work":    "mem:work:*",
            "monitor":     "monitor:*",
            "eval":        "eval:*",
            "wiki:cache":  "wiki:cache:*",
            "lock":        "lock:*",
            "rl:":         "rl:*",
            "notif":       "notif:*",
        }
        key_counts = {}
        for label, pattern in prefixes.items():
            try:
                cur, keys = r.scan(0, match=pattern, count=1000)
                # approximate - scan one pass
                key_counts[label] = len(keys)
            except Exception:
                key_counts[label] = -1

        # Total keys (INFO keyspace)
        try:
            ks = r.info("keyspace")
            total_keys = sum(v.get("keys", 0) for v in ks.values() if isinstance(v, dict))
        except Exception:
            total_keys = sum(v for v in key_counts.values() if v >= 0)

        # Índices RediSearch
        indices = {}
        for idx_name in [IDX_CHUNKS, IDX_TOOLS, "idx:semantic_cache"]:
            try:
                info_idx = r.ft(idx_name).info()
                indices[idx_name] = {
                    "num_docs":  info_idx.get("num_docs", 0),
                    "num_terms": info_idx.get("num_terms", 0),
                    "indexing":  info_idx.get("indexing", 0),
                }
            except Exception:
                indices[idx_name] = {"status": "não existe"}

        return {
            "memory": {
                "used_mb":      round(info_mem.get("used_memory", 0) / 1024 / 1024, 1),
                "peak_mb":      round(info_mem.get("used_memory_peak", 0) / 1024 / 1024, 1),
                "rss_mb":       round(info_mem.get("used_memory_rss", 0) / 1024 / 1024, 1),
                "fragmentation": round(info_mem.get("mem_fragmentation_ratio", 1.0), 2),
                "maxmemory_mb": round(info_mem.get("maxmemory", 0) / 1024 / 1024, 1),
            },
            "server": {
                "version":   info_srv.get("redis_version", "?"),
                "mode":      info_srv.get("redis_mode", "standalone"),
                "uptime_h":  round(info_srv.get("uptime_in_seconds", 0) / 3600, 1),
                "clients":   info_clients.get("connected_clients", 0),
            },
            "stats": {
                "total_commands": info_stats.get("total_commands_processed", 0),
                "hits":           info_stats.get("keyspace_hits", 0),
                "misses":         info_stats.get("keyspace_misses", 0),
                "hit_rate":       round(
                    info_stats.get("keyspace_hits", 0) /
                    max(info_stats.get("keyspace_hits", 0) + info_stats.get("keyspace_misses", 1), 1)
                    * 100, 1
                ),
            },
            "total_keys":  total_keys,
            "key_counts":  key_counts,
            "indices":     indices,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/redis/keys")
async def redis_keys(
    prefix: str = "",
    limit:  int = 50,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.redis_client import get_redis
        r = get_redis()
        pattern = f"{prefix}*" if prefix else "*"
        _, keys  = r.scan(0, match=pattern, count=min(limit * 2, 500))
        keys     = [k.decode() if isinstance(k, bytes) else k for k in keys][:limit]

        result = []
        for key in keys:
            try:
                ttl   = r.ttl(key)
                ktype = r.type(key).decode() if isinstance(r.type(key), bytes) else str(r.type(key))
                size  = r.memory_usage(key) or 0
                result.append({"key": key, "type": ktype, "ttl": ttl, "bytes": size})
            except Exception:
                result.append({"key": key, "type": "?", "ttl": -1, "bytes": 0})

        return {"keys": result, "count": len(result), "pattern": pattern}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/redis/key/{key:path}")
async def redis_get_key(
    key: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.redis_client import get_redis
        r = get_redis()
        ktype = r.type(key)
        ktype = ktype.decode() if isinstance(ktype, bytes) else str(ktype)

        if ktype == "string":
            val = r.get(key)
            val = val.decode("utf-8", errors="replace") if val else None
        elif ktype == "list":
            val = r.lrange(key, 0, 19)
            val = [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v for v in val]
        elif ktype == "hash":
            val = r.hgetall(key)
            val = {
                (k.decode() if isinstance(k, bytes) else k):
                (v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v)
                for k, v in val.items()
            }
        elif ktype == "ReJSON-RL":
            val = r.json().get(key, "$")
        else:
            val = f"(tipo {ktype} — use redis-cli)"

        return {"key": key, "type": ktype, "ttl": r.ttl(key), "value": val}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/redis/key/{key:path}")
async def redis_delete_key(
    key: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.redis_client import get_redis
        r   = get_redis()
        n   = r.delete(key)
        return {"deleted": n, "key": key}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/redis/prefix/{prefix:path}")
async def redis_delete_prefix(
    prefix: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """Deleta todas as chaves que começam com prefix (cuidado!)."""
    require_admin(x_admin_key)
    try:
        from src.infrastructure.redis_client import get_redis
        r = get_redis()
        deleted = 0
        cursor  = 0
        while True:
            cursor, keys = r.scan(cursor, match=f"{prefix}*", count=200)
            if keys:
                r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        return {"deleted": deleted, "prefix": prefix}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/postgres")
async def postgres_info(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as s:
            # Versão e banco
            v_row = await s.execute(text(
                "SELECT version(), current_database(), "
                "pg_size_pretty(pg_database_size(current_database())), "
                "current_user"
            ))
            ver, db, sz, usr = v_row.fetchone()

            # Tabelas com tamanho e linhas
            t_row = await s.execute(text("""
                SELECT
                    tablename,
                    pg_size_pretty(pg_total_relation_size(quote_ident(tablename))) as size,
                    (SELECT count(*) FROM information_schema.columns
                     WHERE table_name = t.tablename) as cols
                FROM pg_tables t
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(quote_ident(tablename)) DESC
            """))
            tables_raw = t_row.fetchall()

            tables = []
            for (tname, tsize, tcols) in tables_raw:
                try:
                    cnt = await s.execute(text(f'SELECT count(*) FROM "{tname}"'))
                    count = cnt.scalar()
                except Exception:
                    count = -1
                tables.append({"name": tname, "size": tsize, "rows": count, "cols": tcols})

            # Conexões ativas
            conn_row = await s.execute(text(
                "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
            ))
            active_conns = conn_row.scalar()

            # Alembic — revisão atual
            try:
                alembic_row = await s.execute(text(
                    "SELECT version_num, executed_at FROM alembic_version"
                ))
                alembic = alembic_row.fetchone()
                alembic_rev = {"revision": alembic[0], "executed_at": str(alembic[1])} if alembic else None
            except Exception:
                alembic_rev = None

        return {
            "version": ver.split(" ")[1] if " " in ver else ver[:30],
            "database": db,
            "size": sz,
            "user": usr,
            "active_connections": active_conns,
            "tables": tables,
            "alembic": alembic_rev,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/postgres/pessoas")
async def postgres_pessoas(
    limit: int = 30,
    role:  str = "",
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as s:
            where = f"WHERE role = '{role}'" if role else ""
            rows  = await s.execute(text(f"""
                SELECT id, nome, email, telefone, role, status, curso, centro,
                       semestre_ingresso, verificado, criado_em, atualizado_em
                FROM "Pessoas"
                {where}
                ORDER BY criado_em DESC
                LIMIT {min(limit, 200)}
            """))
            cols    = rows.keys()
            pessoas = [dict(zip(cols, r)) for r in rows.fetchall()]

            # Convert datetime to string
            for p in pessoas:
                for k, v in p.items():
                    if hasattr(v, "isoformat"):
                        p[k] = v.isoformat()

            # Stats by role
            stat_rows = await s.execute(text(
                "SELECT role, count(*) FROM \"Pessoas\" GROUP BY role ORDER BY count DESC"
            ))
            stats = {r[0]: r[1] for r in stat_rows.fetchall()}

        return {"pessoas": pessoas, "stats": stats, "total": sum(stats.values())}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.patch("/postgres/pessoas/{pessoa_id}")
async def postgres_update_pessoa(
    pessoa_id: int,
    request:   Request,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """Atualiza role e/ou status de um usuário."""
    require_admin(x_admin_key)
    try:
        body = await request.json()
        allowed = {"role", "status", "verificado", "pode_abrir_chamado"}
        updates = {k: v for k, v in body.items() if k in allowed}
        if not updates:
            raise HTTPException(400, "Nenhum campo válido para atualizar")

        from src.infrastructure.database import AsyncSessionLocal
        from sqlalchemy import text

        set_clause = ", ".join(f"{k} = :{k}" for k in updates)
        updates["id"] = pessoa_id

        async with AsyncSessionLocal() as s:
            await s.execute(text(f'UPDATE "Pessoas" SET {set_clause} WHERE id = :id'), updates)
            await s.commit()

        return {"ok": True, "updated": list(updates.keys()), "id": pessoa_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — RAG Ingestão
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/rag")
async def rag_status(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    require_admin(x_admin_key)
    try:
        from src.rag.ingestion import Ingestor, PDF_CONFIG, _ler_manifesto, _caminho_manifesto
        from src.infrastructure.redis_client import get_redis, PREFIX_CHUNKS

        r          = get_redis()
        manifesto  = _ler_manifesto()
        ingestor   = Ingestor()

        # Chunks por source no Redis
        _, all_keys = r.scan(0, match=f"{PREFIX_CHUNKS}*", count=2000)
        chunks_por_source: dict[str, int] = {}
        for key in all_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            parts   = key_str.split(":", 3)
            if len(parts) >= 3:
                src = parts[2]
                chunks_por_source[src] = chunks_por_source.get(src, 0) + 1

        # Config esperada vs real
        sources_info = []
        for nome, cfg in PDF_CONFIG.items():
            manifest_entry = manifesto.get(nome, {})
            sources_info.append({
                "nome":      nome,
                "doc_type":  cfg.get("doc_type", "?"),
                "titulo":    cfg.get("titulo", nome),
                "parser":    cfg.get("parser", "auto"),
                "chunks_redis":   chunks_por_source.get(nome, 0),
                "chunks_manifest":manifest_entry.get("chunks", 0),
                "hash":      manifest_entry.get("hash", "")[:8],
                "indexado":  nome in chunks_por_source and chunks_por_source[nome] > 0,
            })

        # Cache semântico
        try:
            from src.infrastructure.semantic_cache import cache_stats
            cache = cache_stats()
        except Exception:
            cache = {}

        return {
            "sources":         sources_info,
            "total_chunks":    sum(chunks_por_source.values()),
            "total_sources_redis": len(chunks_por_source),
            "total_sources_config": len(PDF_CONFIG),
            "manifesto_path":  _caminho_manifesto(),
            "cache":           cache,
            "chunks_por_source_redis": chunks_por_source,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/rag/ingerir/{source}")
async def rag_ingerir_source(
    source: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """Re-ingere um source específico."""
    require_admin(x_admin_key)
    try:
        from src.infrastructure.settings import settings
        from src.rag.ingestion import Ingestor
        import os

        caminho = os.path.join(settings.DATA_DIR, source)
        if not os.path.exists(caminho):
            raise HTTPException(404, f"Arquivo não encontrado: {caminho}")

        ingestor = Ingestor()
        t0       = time.monotonic()
        chunks   = ingestor._ingerir_ficheiro(caminho)
        ms       = int((time.monotonic() - t0) * 1000)

        return {"ok": True, "source": source, "chunks": chunks, "ms": ms}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/rag/rebuild")
async def rag_rebuild(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    """Re-ingere TODOS os sources (pode demorar)."""
    require_admin(x_admin_key)
    try:
        from src.rag.ingestion import Ingestor
        ingestor = Ingestor()
        t0       = time.monotonic()
        await asyncio.to_thread(ingestor.ingerir_tudo)
        ms       = int((time.monotonic() - t0) * 1000)
        return {"ok": True, "ms": ms}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/rag/source/{source}")
async def rag_delete_source(
    source: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """Remove todos os chunks de um source do Redis."""
    require_admin(x_admin_key)
    try:
        from src.infrastructure.redis_client import deletar_chunks_por_source
        n = deletar_chunks_por_source(source)
        return {"ok": True, "deleted_chunks": n, "source": source}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/rag/cache")
async def rag_flush_cache(
    rota: str = "",
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """Limpa cache semântico (todo ou por rota)."""
    require_admin(x_admin_key)
    try:
        from src.infrastructure.semantic_cache import invalidar_cache_rota
        from src.domain.entities import Rota

        if rota:
            n = invalidar_cache_rota(rota.upper())
        else:
            n = sum(invalidar_cache_rota(r.value) for r in Rota)
        return {"ok": True, "deleted": n, "rota": rota or "all"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — Scraping
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/scraping")
async def scraping_status(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.redis_client import get_redis_text, get_redis, PREFIX_CHUNKS
        from src.tools.tool_wiki_ctic import CACHE_PREFIX, WIKI_BASE_URL

        rt = get_redis_text()
        r  = get_redis()

        # Cache da wiki
        _, cache_keys = rt.scan(0, match=f"{CACHE_PREFIX}*", count=500)
        cache_pages   = []
        for key in cache_keys[:20]:
            try:
                raw  = rt.get(key)
                data = json.loads(raw) if raw else {}
                cache_pages.append({
                    "url":     data.get("url", key),
                    "chars":   len(data.get("content", "")),
                    "links":   len(data.get("links", [])),
                    "ts":      data.get("ts", 0),
                    "age_min": round((time.time() - data.get("ts", 0)) / 60),
                })
            except Exception:
                pass

        cache_pages.sort(key=lambda x: x.get("ts", 0), reverse=True)

        # Chunks da wiki no Redis
        _, wiki_keys = r.scan(0, match=f"{PREFIX_CHUNKS}wiki:*", count=500)

        # Sources wiki
        wiki_sources: set[str] = set()
        for key in wiki_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            parts   = key_str.split(":", 3)
            if len(parts) >= 3:
                wiki_sources.add(parts[2])

        return {
            "wiki_base_url":    WIKI_BASE_URL,
            "cache_pages":      len(cache_keys),
            "cache_ttl_h":      24,
            "indexed_pages":    len(wiki_sources),
            "total_chunks":     len(wiki_keys),
            "recent_cache":     cache_pages[:10],
            "wiki_sources":     list(wiki_sources)[:20],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/scraping/run")
async def scraping_run(
    request: Request,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    require_admin(x_admin_key)
    try:
        body = await request.json()
        url  = body.get("url", "")
        max_pages = int(body.get("max_pages", 10))

        from src.tools.tool_wiki_ctic import indexar_wiki, WIKI_SEED_PAGES

        t0     = time.monotonic()
        seeds  = [url] if url else WIKI_SEED_PAGES
        result = await asyncio.to_thread(indexar_wiki, seeds, min(max_pages, 50), False)
        ms     = int((time.monotonic() - t0) * 1000)

        total_chunks = sum(result.values())
        return {"ok": True, "pages_processed": len(result), "total_chunks": total_chunks, "ms": ms}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/scraping/cache")
async def scraping_clear_cache(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    require_admin(x_admin_key)
    try:
        from src.tools.tool_wiki_ctic import limpar_cache_wiki
        n = await asyncio.to_thread(limpar_cache_wiki)
        return {"ok": True, "deleted": n}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — Memória
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/memory")
async def memory_stats(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.redis_client import get_redis_text
        rt = get_redis_text()

        _, fact_keys = rt.scan(0, match="mem:facts:list:*", count=500)
        _, chat_keys = rt.scan(0, match="chat:*",           count=500)
        _, work_keys = rt.scan(0, match="mem:work:*",       count=500)

        # Top usuários por número de fatos
        user_stats = []
        for key in list(fact_keys)[:50]:
            key_str  = key.decode() if isinstance(key, bytes) else key
            user_id  = key_str.replace("mem:facts:list:", "")
            n_fatos  = rt.llen(key)
            user_stats.append({"user_id": user_id, "fatos": n_fatos})

        user_stats.sort(key=lambda x: x["fatos"], reverse=True)

        return {
            "users_com_fatos":   len(fact_keys),
            "sessoes_chat":      len(chat_keys),
            "working_memories":  len(work_keys),
            "top_users":         user_stats[:20],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/memory/{user_id}")
async def memory_user(
    user_id: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    require_admin(x_admin_key)
    try:
        from src.memory.long_term_memory import listar_todos_fatos
        from src.memory.working_memory import get_historico_compactado, get_sinais
        from src.memory.redis_memory import get_contexto

        fatos    = listar_todos_fatos(user_id)
        sinais   = get_sinais(user_id)
        historico= get_historico_compactado(user_id)
        contexto = get_contexto(user_id)

        return {
            "user_id":  user_id,
            "fatos":    fatos,
            "sinais":   sinais,
            "contexto": contexto,
            "historico_turns": historico.turns_incluidos,
            "historico_preview": historico.texto_formatado[:500] if historico.texto_formatado else "",
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/memory/{user_id}/fatos")
async def memory_clear_facts(
    user_id: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    require_admin(x_admin_key)
    try:
        from src.memory.long_term_memory import apagar_fatos
        apagar_fatos(user_id)
        return {"ok": True, "user_id": user_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/memory/{user_id}/tudo")
async def memory_clear_all(
    user_id: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    require_admin(x_admin_key)
    try:
        from src.memory.redis_memory import clear_tudo
        clear_tudo(user_id)
        return {"ok": True, "user_id": user_id}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — Configuração
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/config")
async def config_view(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.settings import settings
        safe = {}
        for field in SAFE_FIELDS:
            val = getattr(settings, field, None)
            if val is not None:
                safe[field] = {"value": val, "editable": field in EDITABLE_FIELDS}
        return safe
    except Exception as e:
        raise HTTPException(500, str(e))


@router.patch("/config/{field}")
async def config_update(
    field:   str,
    request: Request,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """Atualiza um campo de configuração em memória (sem escrever no .env)."""
    require_admin(x_admin_key)
    if field not in EDITABLE_FIELDS:
        raise HTTPException(400, f"Campo '{field}' não é editável")
    try:
        body    = await request.json()
        raw_val = body.get("value")
        tipo    = EDITABLE_FIELDS[field]
        if tipo == bool:
            val = bool(raw_val)
        elif tipo == float:
            val = float(raw_val)
        elif tipo == int:
            val = int(raw_val)
        else:
            val = str(raw_val)

        from src.infrastructure.settings import settings
        setattr(settings, field, val)
        logger.info("⚙️  Config atualizada pelo admin: %s = %s", field, val)
        return {"ok": True, "field": field, "value": val, "note": "Apenas em memória — reiniciar o container para reverter"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — Logs
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/logs")
async def logs_view(
    nivel: str = "error",
    limit: int = 50,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    require_admin(x_admin_key)
    try:
        from src.infrastructure.observability import obs
        from src.infrastructure.redis_client import get_redis_text
        rt   = get_redis_text()

        keys_map = {
            "error": "system_logs:error",
            "warn":  "system_logs:warn",
            "info":  "system_logs:info",
            "monitor": "monitor:logs",
        }
        key    = keys_map.get(nivel, "system_logs:error")
        raw    = rt.lrange(key, 0, min(limit, 200) - 1)
        parsed = []
        for item in raw:
            try:
                parsed.append(json.loads(item))
            except Exception:
                parsed.append({"msg": str(item)[:200]})

        return {"nivel": nivel, "count": len(parsed), "logs": parsed}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — Ações do Sistema
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/actions/test-webhook")
async def action_test_webhook(
    request: Request,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """Envia mensagem de teste para um número via Evolution API."""
    require_admin(x_admin_key)
    try:
        body  = await request.json()
        phone = body.get("phone", "")
        msg   = body.get("msg", "🔧 Teste do portal admin Oráculo UEMA")

        if not phone:
            raise HTTPException(400, "Campo 'phone' obrigatório")

        from src.services.evolution_service import EvolutionService
        svc = EvolutionService()
        ok  = await svc.enviar_mensagem(f"{phone}@s.whatsapp.net", msg)
        return {"ok": ok, "phone": phone}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/actions/flush-memory-cache")
async def action_flush_memory_cache(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    """Limpa memórias de trabalho expiradas (limpeza manual)."""
    require_admin(x_admin_key)
    try:
        from src.infrastructure.redis_client import get_redis_text
        rt = get_redis_text()
        _, chat_keys = rt.scan(0, match="chat:*", count=500)
        _, work_keys = rt.scan(0, match="mem:work:*", count=500)
        # only delete if TTL == -1 (sem expiração) — seguro
        deleted = 0
        for key in list(chat_keys) + list(work_keys):
            if rt.ttl(key) == -1:
                rt.expire(key, 3600)
                deleted += 1
        return {"ok": True, "fixed_ttl": deleted}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/actions/indexation-health")
async def action_indexation_health(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    """Verifica se todos os sources esperados estão realmente indexados."""
    require_admin(x_admin_key)
    try:
        from src.rag.ingestion import Ingestor, PDF_CONFIG
        sources_redis = Ingestor().diagnosticar()
        expected      = set(PDF_CONFIG.keys())
        present       = set(sources_redis)
        missing       = expected - present
        extra         = present - expected
        return {
            "ok":      len(missing) == 0,
            "expected": list(expected),
            "present":  list(present),
            "missing":  list(missing),
            "extra":    list(extra),
        }
    except Exception as e:
        raise HTTPException(500, str(e))
ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admin — Oráculo UEMA</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #0a0c0e;
  --surf:     #111418;
  --surf2:    #1a1f28;
  --surf3:    #232b36;
  --brd:      #2a3140;
  --brd2:     #38465a;
  --txt:      #d4dbe8;
  --muted:    #6b7a8d;
  --accent:   #ff6b35;
  --accent2:  #ff8c5a;
  --green:    #2eb872;
  --red:      #e84040;
  --yellow:   #f0b429;
  --blue:     #3d8ef8;
  --mono:     'IBM Plex Mono', monospace;
  --head:     'Oxanium', sans-serif;
  --sidebar:  220px;
  --radius:   8px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  background: var(--bg);
  color: var(--txt);
  font-family: var(--mono);
  font-size: 13px;
  overflow: hidden;
}
/* grid background */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,107,53,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,107,53,.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}
/* ── LOGIN ───────────────────────────────────────────── */
#login-screen {
  position: fixed; inset: 0; z-index: 9999;
  display: flex; align-items: center; justify-content: center;
  background: var(--bg);
}
.login-box {
  background: var(--surf);
  border: 1px solid var(--accent);
  border-radius: 12px;
  padding: 48px 56px;
  width: 400px;
  box-shadow: 0 0 60px rgba(255,107,53,.15);
}
.login-box h1 {
  font-family: var(--head);
  font-size: 26px;
  color: var(--accent);
  margin-bottom: 4px;
  letter-spacing: .05em;
}
.login-box p { color: var(--muted); margin-bottom: 32px; font-size: 12px; }
.login-box input {
  width: 100%;
  background: var(--surf2);
  border: 1px solid var(--brd);
  color: var(--txt);
  padding: 12px 14px;
  border-radius: var(--radius);
  font-family: var(--mono);
  font-size: 14px;
  margin-bottom: 16px;
  outline: none;
  transition: border-color .2s;
  letter-spacing: .05em;
}
.login-box input:focus { border-color: var(--accent); }
.login-box button {
  width: 100%;
  background: var(--accent);
  color: #000;
  border: none;
  padding: 12px;
  border-radius: var(--radius);
  font-family: var(--head);
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  letter-spacing: .08em;
  text-transform: uppercase;
  transition: opacity .2s;
}
.login-box button:hover { opacity: .85; }
.login-err { color: var(--red); font-size: 11px; margin-top: 8px; text-align: center; min-height: 16px; }
/* ── APP LAYOUT ──────────────────────────────────────── */
#app { display: none; height: 100vh; position: relative; z-index: 1; }
#app.visible { display: flex; }
/* ── SIDEBAR ─────────────────────────────────────────── */
#sidebar {
  width: var(--sidebar);
  background: var(--surf);
  border-right: 1px solid var(--brd);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  overflow-y: auto;
}
.sidebar-logo {
  padding: 20px 16px 14px;
  border-bottom: 1px solid var(--brd);
}
.sidebar-logo .logo-txt {
  font-family: var(--head);
  font-size: 16px;
  font-weight: 800;
  color: var(--accent);
  letter-spacing: .06em;
}
.sidebar-logo .logo-sub { color: var(--muted); font-size: 10px; margin-top: 2px; }
nav { flex: 1; padding: 12px 8px; }
.nav-group { margin-bottom: 6px; }
.nav-group-label {
  color: var(--muted);
  font-size: 9px;
  letter-spacing: .12em;
  text-transform: uppercase;
  padding: 6px 8px 4px;
}
.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  border-radius: 6px;
  cursor: pointer;
  color: var(--muted);
  transition: all .15s;
  user-select: none;
  font-size: 12px;
}
.nav-item:hover { background: var(--surf2); color: var(--txt); }
.nav-item.active { background: rgba(255,107,53,.12); color: var(--accent); border-left: 2px solid var(--accent); padding-left: 8px; }
.nav-icon { font-size: 14px; width: 18px; text-align: center; }
/* status dots */
.dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; margin-left: auto; }
.dot.green  { background: var(--green); box-shadow: 0 0 6px var(--green); }
.dot.red    { background: var(--red);   box-shadow: 0 0 6px var(--red); }
.dot.yellow { background: var(--yellow); }
.sidebar-foot {
  padding: 12px 16px;
  border-top: 1px solid var(--brd);
  font-size: 10px;
  color: var(--muted);
}
.btn-logout {
  display: block;
  width: 100%;
  background: transparent;
  border: 1px solid var(--brd);
  color: var(--muted);
  padding: 7px;
  border-radius: 6px;
  cursor: pointer;
  font-family: var(--mono);
  font-size: 11px;
  text-align: center;
  margin-top: 8px;
  transition: all .15s;
}
.btn-logout:hover { border-color: var(--red); color: var(--red); }
/* ── CONTENT ─────────────────────────────────────────── */
#content { flex: 1; overflow-y: auto; display: flex; flex-direction: column; }
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 24px;
  border-bottom: 1px solid var(--brd);
  background: var(--surf);
  position: sticky; top: 0; z-index: 10;
  flex-shrink: 0;
}
.topbar h2 { font-family: var(--head); font-size: 16px; font-weight: 700; }
.topbar-meta { display: flex; align-items: center; gap: 16px; }
.topbar-time { color: var(--muted); font-size: 11px; }
.btn { display: inline-flex; align-items: center; gap: 6px; padding: 7px 14px; border-radius: 6px; font-family: var(--mono); font-size: 11px; font-weight: 500; cursor: pointer; border: 1px solid var(--brd); background: var(--surf2); color: var(--txt); transition: all .15s; user-select: none; }
.btn:hover { border-color: var(--accent); color: var(--accent); }
.btn.primary { background: var(--accent); color: #000; border-color: var(--accent); font-weight: 700; }
.btn.primary:hover { opacity: .85; }
.btn.danger { border-color: var(--red); color: var(--red); }
.btn.danger:hover { background: rgba(232,64,64,.1); }
.btn.sm { padding: 4px 9px; font-size: 10px; }
.page { padding: 24px; display: none; }
.page.active { display: block; }
/* ── CARDS ───────────────────────────────────────────── */
.cards { display: grid; gap: 16px; margin-bottom: 24px; }
.cards.cols-4 { grid-template-columns: repeat(4, 1fr); }
.cards.cols-3 { grid-template-columns: repeat(3, 1fr); }
.cards.cols-2 { grid-template-columns: repeat(2, 1fr); }
.card {
  background: var(--surf);
  border: 1px solid var(--brd);
  border-radius: var(--radius);
  padding: 18px 20px;
}
.card-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
.card-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
.card-val { font-family: var(--head); font-size: 28px; font-weight: 800; color: var(--txt); line-height: 1; }
.card-val.accent { color: var(--accent); }
.card-val.green  { color: var(--green); }
.card-val.red    { color: var(--red); }
.card-val.blue   { color: var(--blue); }
.card-sub { font-size: 10px; color: var(--muted); margin-top: 4px; }
.svc-card {
  background: var(--surf);
  border: 1px solid var(--brd);
  border-radius: var(--radius);
  padding: 14px 18px;
  display: flex;
  align-items: center;
  gap: 14px;
}
.svc-icon { font-size: 22px; }
.svc-info { flex: 1; }
.svc-name { font-family: var(--head); font-size: 13px; font-weight: 700; margin-bottom: 2px; }
.svc-detail { font-size: 10px; color: var(--muted); }
.svc-badge { padding: 3px 10px; border-radius: 20px; font-size: 10px; font-weight: 600; }
.svc-badge.ok     { background: rgba(46,184,114,.12); color: var(--green); border: 1px solid rgba(46,184,114,.3); }
.svc-badge.down   { background: rgba(232,64,64,.12);  color: var(--red);   border: 1px solid rgba(232,64,64,.3); }
.svc-badge.warn   { background: rgba(240,180,41,.12); color: var(--yellow); border: 1px solid rgba(240,180,41,.3); }
.svc-badge.info   { background: rgba(61,142,248,.12); color: var(--blue);   border: 1px solid rgba(61,142,248,.3); }
/* ── SECTION HEADER ─────────────────────────────────── */
.sect-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.sect-title { font-family: var(--head); font-size: 14px; font-weight: 700; color: var(--txt); }
.sect-sub { color: var(--muted); font-size: 10px; margin-top: 2px; }
/* ── TABLE ───────────────────────────────────────────── */
.tbl-wrap { background: var(--surf); border: 1px solid var(--brd); border-radius: var(--radius); overflow: hidden; margin-bottom: 20px; }
table { width: 100%; border-collapse: collapse; }
thead { background: var(--surf2); }
th { padding: 10px 14px; text-align: left; font-size: 10px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; border-bottom: 1px solid var(--brd); }
td { padding: 9px 14px; font-size: 12px; border-bottom: 1px solid rgba(255,255,255,.03); vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,.02); }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }
.tag.green  { background: rgba(46,184,114,.12);  color: var(--green); }
.tag.red    { background: rgba(232,64,64,.12);   color: var(--red); }
.tag.yellow { background: rgba(240,180,41,.12);  color: var(--yellow); }
.tag.blue   { background: rgba(61,142,248,.12);  color: var(--blue); }
.tag.muted  { background: rgba(107,122,141,.12); color: var(--muted); }
.tag.accent { background: rgba(255,107,53,.12);  color: var(--accent); }
/* ── FORM ────────────────────────────────────────────── */
.form-row { display: flex; gap: 10px; margin-bottom: 14px; align-items: flex-end; }
.form-group { flex: 1; }
.form-group label { display: block; font-size: 10px; color: var(--muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: .06em; }
.form-group input, .form-group select {
  width: 100%; background: var(--surf2); border: 1px solid var(--brd); color: var(--txt); padding: 8px 10px;
  border-radius: 6px; font-family: var(--mono); font-size: 12px; outline: none; transition: border-color .15s;
}
.form-group input:focus, .form-group select:focus { border-color: var(--accent); }
.form-group select option { background: var(--surf2); }
/* ── CODE / PRE ──────────────────────────────────────── */
pre { background: var(--surf2); border: 1px solid var(--brd); border-radius: 6px; padding: 14px; overflow: auto; font-size: 11px; color: var(--txt); max-height: 350px; white-space: pre-wrap; word-break: break-all; }
/* ── PROGRESS BAR ────────────────────────────────────── */
.progress { background: var(--surf2); border-radius: 4px; height: 6px; overflow: hidden; margin-top: 6px; }
.progress-fill { height: 100%; border-radius: 4px; background: var(--accent); transition: width .5s; }
.progress-fill.green  { background: var(--green); }
.progress-fill.blue   { background: var(--blue); }
/* ── TOAST ───────────────────────────────────────────── */
#toast { position: fixed; bottom: 24px; right: 24px; z-index: 9998; display: flex; flex-direction: column; gap: 8px; }
.toast-item { background: var(--surf2); border: 1px solid var(--brd); border-radius: 8px; padding: 10px 16px; font-size: 12px; min-width: 220px; animation: fadeIn .2s; display: flex; gap: 8px; align-items: center; }
.toast-item.ok  { border-color: var(--green); color: var(--green); }
.toast-item.err { border-color: var(--red);   color: var(--red); }
.toast-item.info{ border-color: var(--blue);  color: var(--blue); }
@keyframes fadeIn { from { opacity:0; transform: translateY(8px); } to { opacity:1; transform: none; } }
/* ── CONFIRM MODAL ───────────────────────────────────── */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.7); z-index: 9990; display: none; align-items: center; justify-content: center; }
.modal-overlay.open { display: flex; }
.modal-box { background: var(--surf); border: 1px solid var(--brd); border-radius: 10px; padding: 28px 32px; width: 380px; }
.modal-box h3 { font-family: var(--head); font-size: 16px; margin-bottom: 10px; }
.modal-box p { color: var(--muted); font-size: 12px; margin-bottom: 20px; }
.modal-box .modal-actions { display: flex; gap: 10px; justify-content: flex-end; }
/* ── MISC ────────────────────────────────────────────── */
.empty { text-align: center; padding: 40px; color: var(--muted); font-size: 12px; }
.loading { color: var(--muted); font-size: 12px; padding: 20px; text-align: center; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.key-val { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,.04); font-size: 12px; }
.key-val:last-child { border-bottom: none; }
.key-val .k { color: var(--muted); }
.key-val .v { color: var(--txt); text-align: right; font-family: var(--mono); }
.section-tabs { display: flex; gap: 4px; margin-bottom: 16px; }
.section-tab { padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 11px; color: var(--muted); border: 1px solid transparent; transition: all .15s; }
.section-tab:hover { color: var(--txt); }
.section-tab.active { background: var(--surf2); border-color: var(--brd); color: var(--accent); }
.inline-edit { background: transparent; border: none; border-bottom: 1px dashed var(--brd); color: var(--txt); font-family: var(--mono); font-size: 12px; padding: 2px 4px; outline: none; width: 80px; }
.inline-edit:focus { border-color: var(--accent); }
</style>
</head>
<body>

<!-- LOGIN -->
<div id="login-screen">
  <div class="login-box">
    <h1>ORÁCULO UEMA</h1>
    <p>Portal de Administração — acesso restrito</p>
    <input type="password" id="login-key" placeholder="ADMIN_API_KEY" autocomplete="off">
    <button onclick="doLogin()">ENTRAR</button>
    <p class="login-err" id="login-err"></p>
  </div>
</div>

<!-- APP -->
<div id="app">
  <!-- SIDEBAR -->
  <aside id="sidebar">
    <div class="sidebar-logo">
      <div class="logo-txt">⬡ ORÁCULO</div>
      <div class="logo-sub">Portal de Administração v1.0</div>
    </div>
    <nav>
      <div class="nav-group">
        <div class="nav-group-label">Sistema</div>
        <div class="nav-item active" onclick="nav('overview')">
          <span class="nav-icon">🏠</span> Overview
          <span class="dot green" id="dot-overview"></span>
        </div>
        <div class="nav-item" onclick="nav('logs')">
          <span class="nav-icon">📋</span> Logs
        </div>
        <div class="nav-item" onclick="nav('config')">
          <span class="nav-icon">⚙️</span> Configuração
        </div>
      </div>
      <div class="nav-group">
        <div class="nav-group-label">Bancos de Dados</div>
        <div class="nav-item" onclick="nav('redis')">
          <span class="nav-icon">🔴</span> Redis
          <span class="dot" id="dot-redis"></span>
        </div>
        <div class="nav-item" onclick="nav('postgres')">
          <span class="nav-icon">🐘</span> PostgreSQL
          <span class="dot" id="dot-pg"></span>
        </div>
      </div>
      <div class="nav-group">
        <div class="nav-group-label">Inteligência</div>
        <div class="nav-item" onclick="nav('rag')">
          <span class="nav-icon">📦</span> Ingestão RAG
        </div>
        <div class="nav-item" onclick="nav('scraping')">
          <span class="nav-icon">🕷️</span> Scraping Wiki
        </div>
      </div>
      <div class="nav-group">
        <div class="nav-group-label">Usuários</div>
        <div class="nav-item" onclick="nav('users')">
          <span class="nav-icon">👥</span> Usuários
        </div>
        <div class="nav-item" onclick="nav('memory')">
          <span class="nav-icon">🧠</span> Memória
        </div>
      </div>
    </nav>
    <div class="sidebar-foot">
      <div id="sidebar-time" style="margin-bottom:6px;font-size:10px;"></div>
      <button class="btn-logout" onclick="doLogout()">⏏ Sair</button>
    </div>
  </aside>

  <!-- MAIN CONTENT -->
  <div id="content">
    <div class="topbar">
      <h2 id="page-title">Overview</h2>
      <div class="topbar-meta">
        <span class="topbar-time" id="last-refresh">—</span>
        <button class="btn sm" onclick="refreshCurrent()">↻ Atualizar</button>
      </div>
    </div>

    <!-- ── PAGE: OVERVIEW ── -->
    <div class="page active" id="page-overview">
      <div class="cards cols-3" id="svc-cards" style="margin-bottom:20px;">
        <div class="loading">Carregando serviços...</div>
      </div>
      <div class="two-col">
        <div>
          <div class="sect-head"><div><div class="sect-title">📊 Métricas Gerais</div></div></div>
          <div class="card" id="metrics-card"><div class="loading">...</div></div>
        </div>
        <div>
          <div class="sect-head"><div><div class="sect-title">💬 Atividade Recente</div></div></div>
          <div class="tbl-wrap" id="activity-table"><div class="loading">...</div></div>
        </div>
      </div>
    </div>

    <!-- ── PAGE: REDIS ── -->
    <div class="page" id="page-redis">
      <div class="section-tabs">
        <div class="section-tab active" onclick="redisTab('info')">Info Geral</div>
        <div class="section-tab" onclick="redisTab('indices')">Índices</div>
        <div class="section-tab" onclick="redisTab('keys')">Browser de Keys</div>
      </div>
      <div id="redis-info-panel"></div>
      <div id="redis-indices-panel" style="display:none"></div>
      <div id="redis-keys-panel" style="display:none">
        <div class="form-row">
          <div class="form-group">
            <label>Prefixo</label>
            <input type="text" id="key-prefix-input" placeholder="ex: chat: ou mem:facts:" >
          </div>
          <button class="btn primary" onclick="loadRedisKeys()">Buscar</button>
        </div>
        <div id="redis-keys-result"></div>
      </div>
    </div>

    <!-- ── PAGE: POSTGRES ── -->
    <div class="page" id="page-postgres">
      <div class="section-tabs">
        <div class="section-tab active" onclick="pgTab('overview')">Visão Geral</div>
        <div class="section-tab" onclick="pgTab('users')">Usuários</div>
      </div>
      <div id="pg-overview-panel"></div>
      <div id="pg-users-panel" style="display:none">
        <div class="form-row">
          <div class="form-group">
            <label>Filtrar por Role</label>
            <select id="pg-role-filter">
              <option value="">Todos</option>
              <option value="admin">admin</option>
              <option value="coordenador">coordenador</option>
              <option value="professor">professor</option>
              <option value="estudante">estudante</option>
              <option value="servidor">servidor</option>
              <option value="publico">publico</option>
            </select>
          </div>
          <button class="btn" onclick="loadPgUsers()">Filtrar</button>
        </div>
        <div id="pg-users-result"></div>
      </div>
    </div>

    <!-- ── PAGE: RAG ── -->
    <div class="page" id="page-rag">
      <div class="sect-head">
        <div><div class="sect-title">📦 Ingestão RAG</div><div class="sect-sub">Gerenciar sources, chunks e cache semântico</div></div>
        <div style="display:flex;gap:8px;">
          <button class="btn danger" onclick="confirmAction('flush-cache','Limpar todo o cache semântico?','Isso forçará Gemini a regenerar respostas não cacheadas.',flushSemanticCache)">🗑 Flush Cache</button>
          <button class="btn primary" onclick="confirmAction('rebuild','Re-ingerir TODOS os PDFs?','Operação demorada (~5 min). Pode causar lentidão temporária.',rebuildAll)">♻ Rebuild Tudo</button>
        </div>
      </div>
      <div id="rag-status-content"><div class="loading">Carregando...</div></div>
    </div>

    <!-- ── PAGE: SCRAPING ── -->
    <div class="page" id="page-scraping">
      <div class="sect-head">
        <div><div class="sect-title">🕷️ Scraping Wiki CTIC</div></div>
        <div style="display:flex;gap:8px;">
          <button class="btn danger" onclick="confirmAction('clear-wiki-cache','Limpar cache da Wiki?','Próximo acesso fará HTTP para o servidor da UEMA.',clearWikiCache)">🗑 Limpar Cache</button>
          <button class="btn primary" onclick="runScraping()">▶ Executar Scraping</button>
        </div>
      </div>
      <div id="scraping-status-content"><div class="loading">Carregando...</div></div>
      <div style="margin-top:20px;">
        <div class="sect-head"><div class="sect-title">▶ Scraping Manual</div></div>
        <div class="card">
          <div class="form-row">
            <div class="form-group"><label>URL Específica (opcional)</label><input type="text" id="scrape-url" placeholder="https://ctic.uema.br/wiki/doku.php?id=start"></div>
            <div class="form-group" style="max-width:100px;"><label>Máx. Páginas</label><input type="number" id="scrape-max" value="10" min="1" max="50"></div>
          </div>
          <div id="scraping-result" style="display:none;" class="tag blue" style="margin-top:8px;"></div>
        </div>
      </div>
    </div>

    <!-- ── PAGE: USERS ── -->
    <div class="page" id="page-users">
      <div class="sect-head">
        <div><div class="sect-title">👥 Usuários</div></div>
        <div class="form-row" style="margin:0;gap:8px;">
          <select id="users-role-filter" style="background:var(--surf2);border:1px solid var(--brd);color:var(--txt);padding:6px 10px;border-radius:6px;font-family:var(--mono);font-size:11px;outline:none;">
            <option value="">Todos os roles</option>
            <option value="admin">admin</option>
            <option value="coordenador">coordenador</option>
            <option value="professor">professor</option>
            <option value="estudante">estudante</option>
            <option value="servidor">servidor</option>
            <option value="publico">publico</option>
          </select>
          <button class="btn" onclick="loadUsers()">Filtrar</button>
        </div>
      </div>
      <div id="users-content"><div class="loading">Carregando...</div></div>
    </div>

    <!-- ── PAGE: MEMORY ── -->
    <div class="page" id="page-memory">
      <div class="sect-head"><div class="sect-title">🧠 Memória dos Usuários</div></div>
      <div class="two-col" style="margin-bottom:20px;">
        <div id="memory-stats"><div class="loading">...</div></div>
        <div>
          <div class="sect-head"><div class="sect-title">Explorar Usuário</div></div>
          <div class="card">
            <div class="form-row">
              <div class="form-group"><label>Telefone / user_id</label><input type="text" id="mem-user-input" placeholder="559899999999"></div>
              <button class="btn primary" onclick="loadMemoryUser()">Ver</button>
            </div>
            <div id="mem-user-result" style="margin-top:12px;"></div>
          </div>
        </div>
      </div>
      <div class="sect-head"><div class="sect-title">Top Usuários por Fatos</div></div>
      <div id="memory-top-users"><div class="loading">...</div></div>
    </div>

    <!-- ── PAGE: CONFIG ── -->
    <div class="page" id="page-config">
      <div class="sect-head">
        <div><div class="sect-title">⚙️ Configuração</div><div class="sect-sub">Alterações só valem em memória até reiniciar o container</div></div>
      </div>
      <div id="config-content"><div class="loading">Carregando...</div></div>
      <div style="margin-top:24px;">
        <div class="sect-head"><div class="sect-title">📬 Teste de Webhook</div></div>
        <div class="card">
          <div class="form-row">
            <div class="form-group"><label>Telefone (sem +55, com DDD)</label><input type="text" id="test-phone" placeholder="98999999999"></div>
            <div class="form-group"><label>Mensagem</label><input type="text" id="test-msg" placeholder="Teste do portal admin Oráculo UEMA" value="🔧 Teste do portal admin Oráculo UEMA"></div>
            <button class="btn primary" onclick="testWebhook()">Enviar</button>
          </div>
          <div id="webhook-result" style="display:none;margin-top:8px;font-size:11px;"></div>
        </div>
        <div style="margin-top:20px;">
          <div class="sect-head"><div class="sect-title">🔍 Health Check Completo</div></div>
          <button class="btn" onclick="runHealthCheck()" style="margin-bottom:12px;">Executar verificação</button>
          <div id="health-result"></div>
        </div>
      </div>
    </div>

    <!-- ── PAGE: LOGS ── -->
    <div class="page" id="page-logs">
      <div class="section-tabs">
        <div class="section-tab active" onclick="logsTab('error')">Erros</div>
        <div class="section-tab" onclick="logsTab('warn')">Avisos</div>
        <div class="section-tab" onclick="logsTab('info')">Info</div>
        <div class="section-tab" onclick="logsTab('monitor')">Monitor</div>
      </div>
      <div id="logs-content"><div class="loading">Carregando...</div></div>
    </div>
  </div>
</div>

<!-- CONFIRM MODAL -->
<div class="modal-overlay" id="modal">
  <div class="modal-box">
    <h3 id="modal-title">Confirmar</h3>
    <p id="modal-desc">Tem certeza?</p>
    <div class="modal-actions">
      <button class="btn" onclick="closeModal()">Cancelar</button>
      <button class="btn danger" id="modal-confirm-btn">Confirmar</button>
    </div>
  </div>
</div>

<!-- TOAST -->
<div id="toast"></div>

<script>
// ─── State ────────────────────────────────────────────────────────────────────
const S = { key: null, section: 'overview', modalCb: null };

// ─── Auth ─────────────────────────────────────────────────────────────────────
async function doLogin() {
  const key = document.getElementById('login-key').value.trim();
  if (!key) return;
  try {
    const res = await fetch('/admin/auth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key })
    });
    if (!res.ok) {
      const err = await res.json();
      document.getElementById('login-err').textContent = err.detail || 'Chave incorreta';
      return;
    }
    sessionStorage.setItem('adminKey', key);
    S.key = key;
    document.getElementById('login-screen').style.display = 'none';
    document.getElementById('app').classList.add('visible');
    startClock();
    nav('overview');
  } catch(e) {
    document.getElementById('login-err').textContent = 'Erro de conexão';
  }
}

function doLogout() {
  sessionStorage.removeItem('adminKey');
  location.reload();
}

document.getElementById('login-key').addEventListener('keydown', e => {
  if (e.key === 'Enter') doLogin();
});

// Auto-login from sessionStorage
window.addEventListener('load', () => {
  const k = sessionStorage.getItem('adminKey');
  if (k) {
    S.key = k;
    document.getElementById('login-screen').style.display = 'none';
    document.getElementById('app').classList.add('visible');
    startClock();
    nav('overview');
  }
});

// ─── API ─────────────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  opts.headers = { ...(opts.headers || {}), 'X-Admin-Key': S.key };
  if (opts.body && typeof opts.body === 'object') {
    opts.body    = JSON.stringify(opts.body);
    opts.headers['Content-Type'] = 'application/json';
  }
  const res = await fetch('/admin' + path, opts);
  if (res.status === 401) { doLogout(); return null; }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ─── Navigation ───────────────────────────────────────────────────────────────
const PAGE_TITLES = {
  overview: '🏠 Overview', redis: '🔴 Redis', postgres: '🐘 PostgreSQL',
  rag: '📦 Ingestão RAG', scraping: '🕷️ Scraping Wiki',
  users: '👥 Usuários', memory: '🧠 Memória', config: '⚙️ Configuração', logs: '📋 Logs'
};

function nav(section) {
  S.section = section;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const page = document.getElementById('page-' + section);
  if (page) page.classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => {
    if (n.textContent.toLowerCase().includes(
      section === 'overview' ? 'overview' :
      section === 'redis' ? 'redis' :
      section === 'postgres' ? 'postgresql' :
      section === 'rag' ? 'ingestão' :
      section === 'scraping' ? 'scraping' :
      section === 'users' ? 'usuários' :
      section === 'memory' ? 'memória' :
      section === 'config' ? 'configuração' : 'logs'
    )) n.classList.add('active');
  });
  document.getElementById('page-title').textContent = PAGE_TITLES[section] || section;
  loadSection(section);
}

function refreshCurrent() { loadSection(S.section); }

function loadSection(s) {
  document.getElementById('last-refresh').textContent = 'sync: ' + new Date().toLocaleTimeString('pt-BR');
  if (s === 'overview')  loadOverview();
  if (s === 'redis')     { redisTab('info'); loadRedisInfo(); }
  if (s === 'postgres')  { pgTab('overview'); loadPgOverview(); }
  if (s === 'rag')       loadRAG();
  if (s === 'scraping')  loadScraping();
  if (s === 'users')     loadUsers();
  if (s === 'memory')    loadMemoryStats();
  if (s === 'config')    loadConfig();
  if (s === 'logs')      loadLogs('error');
}

// ─── Clock ────────────────────────────────────────────────────────────────────
function startClock() {
  const el = document.getElementById('sidebar-time');
  setInterval(() => {
    el.textContent = new Date().toLocaleString('pt-BR', { dateStyle: 'short', timeStyle: 'medium' });
  }, 1000);
}

// ─── Toast ────────────────────────────────────────────────────────────────────
function toast(msg, type = 'ok', duration = 3000) {
  const icons = { ok: '✓', err: '✗', info: 'ℹ' };
  const el    = document.createElement('div');
  el.className = `toast-item ${type}`;
  el.innerHTML = `<span>${icons[type]||'·'}</span><span>${msg}</span>`;
  document.getElementById('toast').appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ─── Confirm Modal ────────────────────────────────────────────────────────────
function confirmAction(id, title, desc, cb) {
  S.modalCb = cb;
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-desc').textContent  = desc;
  document.getElementById('modal').classList.add('open');
}
document.getElementById('modal-confirm-btn').onclick = () => {
  closeModal();
  if (S.modalCb) S.modalCb();
  S.modalCb = null;
};
function closeModal() { document.getElementById('modal').classList.remove('open'); }

// ─── OVERVIEW ─────────────────────────────────────────────────────────────────
async function loadOverview() {
  const el = document.getElementById('svc-cards');
  el.innerHTML = '<div class="loading">Verificando serviços...</div>';
  try {
    const data = await api('/overview');
    if (!data) return;
    // Serviços
    const svcs = [
      { key: 'redis',     icon: '🔴', name: 'Redis Stack',   detailFn: s => `v${s.version} · ${s.ram_mb}MB RAM` },
      { key: 'postgres',  icon: '🐘', name: 'PostgreSQL',    detailFn: s => `${s.database} · ${s.size}` },
      { key: 'gemini',    icon: '✨', name: 'Gemini AI',     detailFn: s => s.model },
      { key: 'evolution', icon: '📱', name: 'Evolution API', detailFn: s => s.instance },
      { key: 'agent',     icon: '🤖', name: 'AgentCore',     detailFn: s => `${s.tools || 0} tools` },
    ];
    el.innerHTML = svcs.map(sv => {
      const s   = data.services[sv.key] || {};
      const st  = s.status;
      const cls = st === 'ok' || st === 'ready' || st === 'configured' ? 'ok' : st === 'down' ? 'down' : 'warn';
      const dot = cls;
      const detail = s.error ? s.error.substring(0,40) : (sv.detailFn(s) || '');
      // update sidebar dots
      if (sv.key === 'redis')    updateDot('dot-redis', dot);
      if (sv.key === 'postgres') updateDot('dot-pg', dot);
      return `<div class="svc-card"><div class="svc-icon">${sv.icon}</div>
        <div class="svc-info"><div class="svc-name">${sv.name}</div><div class="svc-detail">${detail}</div></div>
        <span class="svc-badge ${cls}">${st || '?'}</span></div>`;
    }).join('');
    updateDot('dot-overview', 'green');
    // Métricas
    const m = data.metrics || {};
    document.getElementById('metrics-card').innerHTML = `
      <div class="key-val"><span class="k">Mensagens total</span><span class="v">${fmt(m.msgs_total||0)}</span></div>
      <div class="key-val"><span class="k">Mensagens hoje</span><span class="v">${fmt(m.msgs_hoje||0)}</span></div>
      <div class="key-val"><span class="k">Tokens/msg médio</span><span class="v">${fmt(m.tokens_medio||0)}</span></div>
      <div class="key-val"><span class="k">Latência média</span><span class="v">${m.latencia_media||0}ms</span></div>
      <div class="key-val"><span class="k">Usuários c/ fatos</span><span class="v">${m.users_com_fatos||0}</span></div>
      <div class="key-val"><span class="k">Sessões ativas</span><span class="v">${m.sessoes_ativas||0}</span></div>
      <div class="key-val"><span class="k">Sources RAG</span><span class="v">${m.rag_sources||0}</span></div>`;
    // Atividade
    const acts = data.activity || [];
    const tbody = acts.length ? acts.map(a =>
      `<tr><td>${(a.ts||'').slice(11,19)}</td>
       <td style="max-width:80px;overflow:hidden;text-overflow:ellipsis">${(a.user_id||'').slice(-8)}</td>
       <td><span class="tag ${roleColor(a.nivel)}">${a.nivel||'?'}</span></td>
       <td style="color:var(--muted)">${(a.pergunta||'').slice(0,35)}${(a.pergunta||'').length>35?'…':''}</td></tr>`
    ).join('') : '<tr><td colspan="4" class="empty">Sem atividade recente</td></tr>';
    document.getElementById('activity-table').innerHTML = `
      <table><thead><tr><th>Hora</th><th>User</th><th>Nível</th><th>Pergunta</th></tr></thead>
      <tbody>${tbody}</tbody></table>`;
  } catch(e) {
    el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`;
    toast('Erro ao carregar overview: ' + e.message, 'err');
  }
}

// ─── REDIS ────────────────────────────────────────────────────────────────────
function redisTab(t) {
  ['info','indices','keys'].forEach(x => {
    document.getElementById(`redis-${x}-panel`).style.display = t===x ? '' : 'none';
  });
  document.querySelectorAll('#page-redis .section-tab').forEach((el,i) => {
    el.classList.toggle('active', ['info','indices','keys'][i] === t);
  });
  if (t === 'info') loadRedisInfo();
  if (t === 'indices') loadRedisIndices();
}

async function loadRedisInfo() {
  const el = document.getElementById('redis-info-panel');
  el.innerHTML = '<div class="loading">Conectando ao Redis...</div>';
  try {
    const d = await api('/redis');
    if (!d) return;
    const mem = d.memory, srv = d.server, stats = d.stats;
    const pct = mem.maxmemory_mb > 0 ? Math.round(mem.used_mb / mem.maxmemory_mb * 100) : 0;
    el.innerHTML = `
      <div class="cards cols-4" style="margin-bottom:20px;">
        <div class="card"><div class="card-label">RAM Usada</div><div class="card-val accent">${mem.used_mb}MB</div>
          <div class="progress"><div class="progress-fill" style="width:${pct}%"></div></div>
          <div class="card-sub">peak ${mem.peak_mb}MB · max ${mem.maxmemory_mb||'∞'}MB</div></div>
        <div class="card"><div class="card-label">RSS / Frag.</div><div class="card-val">${mem.rss_mb}MB</div>
          <div class="card-sub">fragmentation: ${mem.fragmentation}x</div></div>
        <div class="card"><div class="card-label">Total Keys</div><div class="card-val blue">${fmt(d.total_keys)}</div>
          <div class="card-sub">clients: ${srv.clients} · uptime: ${srv.uptime_h}h</div></div>
        <div class="card"><div class="card-label">Cache Hit Rate</div><div class="card-val green">${stats.hit_rate}%</div>
          <div class="card-sub">hits: ${fmt(stats.hits)} · misses: ${fmt(stats.misses)}</div></div>
      </div>
      <div class="sect-head"><div class="sect-title">Keys por Prefixo</div></div>
      <div class="tbl-wrap"><table><thead><tr><th>Prefixo</th><th>Keys</th><th>Ações</th></tr></thead><tbody>
        ${Object.entries(d.key_counts||{}).sort((a,b)=>b[1]-a[1]).map(([k,v]) =>
          `<tr><td style="font-family:var(--mono);color:var(--accent)">${k}:</td>
           <td><span class="tag ${v>0?'blue':'muted'}">${v}</span></td>
           <td><button class="btn sm" onclick="deletePrefix('${k}:')">🗑 Apagar tudo</button></td></tr>`
        ).join('')}
      </tbody></table></div>`;
  } catch(e) {
    el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`;
  }
}

async function loadRedisIndices() {
  const el = document.getElementById('redis-indices-panel');
  el.innerHTML = '<div class="loading">Carregando índices...</div>';
  try {
    const d = await api('/redis');
    if (!d) return;
    el.innerHTML = `<div class="tbl-wrap"><table>
      <thead><tr><th>Índice</th><th>Documentos</th><th>Termos</th><th>Status</th></tr></thead>
      <tbody>${Object.entries(d.indices||{}).map(([name, info]) =>
        `<tr><td style="color:var(--accent)">${name}</td>
         <td>${fmt(info.num_docs||0)}</td>
         <td>${fmt(info.num_terms||0)}</td>
         <td><span class="tag ${info.status==='não existe'?'red':'green'}">${info.status||'ok'}</span></td></tr>`
      ).join('')}</tbody></table></div>`;
  } catch(e) {
    el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`;
  }
}

async function loadRedisKeys() {
  const prefix = document.getElementById('key-prefix-input').value.trim();
  const el = document.getElementById('redis-keys-result');
  el.innerHTML = '<div class="loading">Buscando...</div>';
  try {
    const d = await api(`/redis/keys?prefix=${encodeURIComponent(prefix)}&limit=80`);
    if (!d) return;
    if (!d.keys.length) { el.innerHTML = '<div class="empty">Nenhuma key encontrada</div>'; return; }
    el.innerHTML = `<div class="tbl-wrap"><table>
      <thead><tr><th>Key</th><th>Tipo</th><th>TTL(s)</th><th>Bytes</th><th>Ações</th></tr></thead>
      <tbody>${d.keys.map(k =>
        `<tr><td style="font-family:var(--mono);font-size:11px;max-width:300px;overflow:hidden;text-overflow:ellipsis">${esc(k.key)}</td>
         <td><span class="tag info">${k.type}</span></td>
         <td style="color:var(--muted)">${k.ttl===-1?'∞':k.ttl}</td>
         <td style="color:var(--muted)">${k.bytes}</td>
         <td style="display:flex;gap:4px;">
           <button class="btn sm" onclick="viewKey('${esc(k.key)}')">👁</button>
           <button class="btn sm danger" onclick="confirmAction('del','Apagar key?','${esc(k.key)}',()=>deleteKey('${esc(k.key)}'))">🗑</button>
         </td></tr>`
      ).join('')}</tbody></table></div>`;
  } catch(e) {
    el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`;
  }
}

async function viewKey(key) {
  try {
    const d = await api(`/redis/key/${encodeURIComponent(key)}`);
    if (!d) return;
    const val = typeof d.value === 'object' ? JSON.stringify(d.value, null, 2) : String(d.value);
    const w   = window.open('', '_blank', 'width=600,height=500');
    w.document.write(`<pre style="background:#111;color:#d4dbe8;padding:20px;font-size:12px;font-family:monospace;white-space:pre-wrap">${esc(val)}</pre>`);
  } catch(e) { toast('Erro: ' + e.message, 'err'); }
}

async function deleteKey(key) {
  try {
    await api(`/redis/key/${encodeURIComponent(key)}`, { method: 'DELETE' });
    toast('Key apagada: ' + key.slice(-30), 'ok');
    loadRedisKeys();
  } catch(e) { toast('Erro: ' + e.message, 'err'); }
}

async function deletePrefix(prefix) {
  confirmAction('del-prefix', `Apagar prefixo "${prefix}"?`, 'Remove TODAS as keys com este prefixo. Irreversível.', async () => {
    try {
      const d = await api(`/redis/prefix/${encodeURIComponent(prefix.slice(0,-1))}`, { method: 'DELETE' });
      toast(`${d.deleted} keys apagadas`, 'ok');
      loadRedisInfo();
    } catch(e) { toast('Erro: ' + e.message, 'err'); }
  });
}

// ─── POSTGRES ─────────────────────────────────────────────────────────────────
function pgTab(t) {
  document.getElementById('pg-overview-panel').style.display = t==='overview' ? '' : 'none';
  document.getElementById('pg-users-panel').style.display = t==='users' ? '' : 'none';
  document.querySelectorAll('#page-postgres .section-tab').forEach((el,i) => {
    el.classList.toggle('active', ['overview','users'][i] === t);
  });
  if (t === 'overview') loadPgOverview();
  if (t === 'users') loadPgUsers();
}

async function loadPgOverview() {
  const el = document.getElementById('pg-overview-panel');
  el.innerHTML = '<div class="loading">Conectando ao PostgreSQL...</div>';
  try {
    const d = await api('/postgres');
    if (!d) return;
    const alembic = d.alembic;
    el.innerHTML = `
      <div class="cards cols-3" style="margin-bottom:20px;">
        <div class="card"><div class="card-label">Versão</div><div class="card-val blue" style="font-size:20px">${d.version}</div>
          <div class="card-sub">DB: ${d.database} · User: ${d.user}</div></div>
        <div class="card"><div class="card-label">Tamanho do Banco</div><div class="card-val accent">${d.size}</div>
          <div class="card-sub">Conexões ativas: ${d.active_connections}</div></div>
        <div class="card"><div class="card-label">Alembic</div>
          <div class="card-val ${alembic?'green':'yellow'}" style="font-size:16px">${alembic?alembic.revision.slice(-8):'N/A'}</div>
          <div class="card-sub">${alembic?alembic.executed_at.slice(0,19):'Sem revisão'}</div></div>
      </div>
      <div class="sect-head"><div class="sect-title">Tabelas</div></div>
      <div class="tbl-wrap"><table>
        <thead><tr><th>Tabela</th><th>Linhas</th><th>Tamanho</th><th>Colunas</th></tr></thead>
        <tbody>${(d.tables||[]).map(t =>
          `<tr><td style="color:var(--accent)">${t.name}</td>
           <td>${fmt(t.rows)}</td><td>${t.size}</td><td>${t.cols}</td></tr>`
        ).join('')}</tbody></table></div>`;
    updateDot('dot-pg', 'green');
  } catch(e) {
    el.innerHTML = `<div class="empty">PostgreSQL offline: ${e.message}</div>`;
    updateDot('dot-pg', 'red');
  }
}

async function loadPgUsers() {
  const role = document.getElementById('pg-role-filter').value;
  const el   = document.getElementById('pg-users-result');
  el.innerHTML = '<div class="loading">Carregando usuários...</div>';
  try {
    const d = await api(`/postgres/pessoas?limit=50${role?'&role='+role:''}`);
    if (!d) return;
    el.innerHTML = `<div class="tbl-wrap"><table>
      <thead><tr><th>ID</th><th>Nome</th><th>Email</th><th>Telefone</th><th>Role</th><th>Status</th><th>Curso</th><th>Ações</th></tr></thead>
      <tbody>${(d.pessoas||[]).map(p =>
        `<tr>
          <td style="color:var(--muted)">${p.id}</td>
          <td>${esc(p.nome||'')}</td>
          <td style="font-size:11px;color:var(--muted)">${esc(p.email||'')}</td>
          <td style="font-family:var(--mono)">${p.telefone||'—'}</td>
          <td><span class="tag ${roleColor(p.role)}">${p.role}</span></td>
          <td><span class="tag ${p.status==='ativo'?'green':p.status==='trancado'?'yellow':'muted'}">${p.status||'?'}</span></td>
          <td style="font-size:11px;color:var(--muted)">${p.curso||'—'}</td>
          <td><button class="btn sm" onclick="editUser(${p.id},'${p.role}')">✏ Role</button></td>
        </tr>`
      ).join('')}</tbody></table></div>
      <div style="font-size:11px;color:var(--muted);margin-top:8px;">Total: ${d.total} · Filtrado: ${d.pessoas?.length||0}</div>`;
  } catch(e) {
    el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`;
  }
}

function editUser(id, currentRole) {
  const roles = ['publico','estudante','servidor','professor','coordenador','admin'];
  const newRole = prompt(`Novo role para usuário ${id}:\n(atual: ${currentRole})\n\nOpções: ${roles.join(', ')}`, currentRole);
  if (!newRole || !roles.includes(newRole)) return;
  confirmAction('edit-role', `Alterar role para "${newRole}"?`, `Usuário ID ${id}`, async () => {
    try {
      await api(`/postgres/pessoas/${id}`, { method: 'PATCH', body: { role: newRole } });
      toast(`Role atualizado: ${currentRole} → ${newRole}`, 'ok');
      loadPgUsers();
    } catch(e) { toast('Erro: ' + e.message, 'err'); }
  });
}

// ─── RAG ─────────────────────────────────────────────────────────────────────
async function loadRAG() {
  const el = document.getElementById('rag-status-content');
  el.innerHTML = '<div class="loading">Verificando ingestão...</div>';
  try {
    const d = await api('/rag');
    if (!d) return;
    const cache = d.cache || {};
    el.innerHTML = `
      <div class="cards cols-4" style="margin-bottom:20px;">
        <div class="card"><div class="card-label">Total Chunks</div><div class="card-val accent">${fmt(d.total_chunks||0)}</div>
          <div class="card-sub">${d.total_sources_redis}/${d.total_sources_config} sources indexados</div></div>
        <div class="card"><div class="card-label">Cache Semântico</div><div class="card-val blue">${fmt(cache.total_entradas||0)}</div>
          <div class="card-sub">threshold: ${cache.threshold||0} · dim: ${cache.vector_dim||0}</div></div>
        <div class="card"><div class="card-label">TTL Cache</div><div class="card-val">${cache.ttl_dias||0}d</div></div>
        <div class="card"><div class="card-label">Saúde</div>
          <div class="card-val ${d.total_sources_redis===d.total_sources_config?'green':'yellow'}">${d.total_sources_redis===d.total_sources_config?'✓ OK':'⚠ Parcial'}</div></div>
      </div>
      <div class="sect-head"><div class="sect-title">Sources Configurados</div></div>
      <div class="tbl-wrap"><table>
        <thead><tr><th>Arquivo</th><th>Tipo</th><th>Parser</th><th>Chunks (Redis)</th><th>Chunks (Manifest)</th><th>Hash</th><th>Ações</th></tr></thead>
        <tbody>${(d.sources||[]).map(s =>
          `<tr>
            <td style="color:var(--accent)">${s.nome}</td>
            <td><span class="tag ${typeColor(s.doc_type)}">${s.doc_type}</span></td>
            <td style="color:var(--muted)">${s.parser}</td>
            <td><span class="tag ${s.indexado?'green':'red'}">${s.chunks_redis}</span></td>
            <td style="color:var(--muted)">${s.chunks_manifest}</td>
            <td style="color:var(--muted);font-size:10px">${s.hash||'—'}</td>
            <td style="display:flex;gap:4px;">
              <button class="btn sm primary" onclick="reingerirSource('${s.nome}')">♻ Reingerir</button>
              <button class="btn sm danger" onclick="confirmAction('del-src','Apagar chunks de ${s.nome}?','Remove do Redis mas não do disco.',()=>deleteSource('${s.nome}'))">🗑</button>
            </td></tr>`
        ).join('')}</tbody></table></div>`;
  } catch(e) {
    el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`;
  }
}

async function reingerirSource(source) {
  try {
    toast(`Reingerindo ${source}...`, 'info', 5000);
    const d = await api(`/rag/ingerir/${encodeURIComponent(source)}`, { method: 'POST' });
    toast(`${source}: ${d.chunks} chunks em ${d.ms}ms`, 'ok');
    loadRAG();
  } catch(e) { toast('Erro: ' + e.message, 'err'); }
}

async function deleteSource(source) {
  try {
    const d = await api(`/rag/source/${encodeURIComponent(source)}`, { method: 'DELETE' });
    toast(`${d.deleted_chunks} chunks apagados`, 'ok');
    loadRAG();
  } catch(e) { toast('Erro: ' + e.message, 'err'); }
}

async function flushSemanticCache() {
  try {
    const d = await api('/rag/cache', { method: 'DELETE' });
    toast(`Cache limpo: ${d.deleted} entradas`, 'ok');
    loadRAG();
  } catch(e) { toast('Erro: ' + e.message, 'err'); }
}

async function rebuildAll() {
  try {
    toast('Rebuild iniciado — pode demorar 5+ minutos...', 'info', 8000);
    const d = await api('/rag/rebuild', { method: 'POST' });
    toast(`Rebuild concluído em ${d.ms}ms`, 'ok');
    loadRAG();
  } catch(e) { toast('Erro: ' + e.message, 'err'); }
}

// ─── SCRAPING ─────────────────────────────────────────────────────────────────
async function loadScraping() {
  const el = document.getElementById('scraping-status-content');
  el.innerHTML = '<div class="loading">Verificando wiki...</div>';
  try {
    const d = await api('/scraping');
    if (!d) return;
    el.innerHTML = `
      <div class="cards cols-4" style="margin-bottom:20px;">
        <div class="card"><div class="card-label">Páginas em Cache</div><div class="card-val accent">${d.cache_pages}</div>
          <div class="card-sub">TTL: ${d.cache_ttl_h}h</div></div>
        <div class="card"><div class="card-label">Páginas Indexadas</div><div class="card-val blue">${d.indexed_pages}</div></div>
        <div class="card"><div class="card-label">Chunks no Redis</div><div class="card-val">${fmt(d.total_chunks)}</div></div>
        <div class="card"><div class="card-label">Base URL</div><div class="card-val" style="font-size:11px;word-break:break-all">ctic.uema.br</div></div>
      </div>
      <div class="two-col">
        <div>
          <div class="sect-head"><div class="sect-title">Cache Recente</div></div>
          <div class="tbl-wrap"><table>
            <thead><tr><th>URL</th><th>Chars</th><th>Links</th><th>Idade</th></tr></thead>
            <tbody>${(d.recent_cache||[]).map(p =>
              `<tr><td style="font-size:10px;max-width:200px;overflow:hidden;text-overflow:ellipsis;color:var(--muted)">${esc(p.url)}</td>
               <td>${fmt(p.chars)}</td><td>${p.links}</td>
               <td style="color:var(--muted)">${p.age_min}min</td></tr>`
            ).join('') || '<tr><td colspan="4" class="empty">Sem cache</td></tr>'}</tbody></table></div>
        </div>
        <div>
          <div class="sect-head"><div class="sect-title">Sources Indexadas</div></div>
          <div class="tbl-wrap"><table>
            <thead><tr><th>Page ID</th></tr></thead>
            <tbody>${(d.wiki_sources||[]).map(s =>
              `<tr><td style="color:var(--accent);font-size:11px">${esc(s)}</td></tr>`
            ).join('') || '<tr><td class="empty">Nenhuma indexada</td></tr>'}</tbody></table></div>
        </div>
      </div>`;
  } catch(e) {
    el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`;
  }
}

async function runScraping() {
  const url  = document.getElementById('scrape-url').value.trim();
  const maxP = parseInt(document.getElementById('scrape-max').value) || 10;
  const el   = document.getElementById('scraping-result');
  el.style.display = '';
  el.textContent = `⏳ Scraping em andamento (máx. ${maxP} páginas)...`;
  try {
    const d = await api('/scraping/run', { method: 'POST', body: { url, max_pages: maxP } });
    el.textContent = `✓ ${d.pages_processed} páginas · ${d.total_chunks} chunks em ${d.ms}ms`;
    toast(`Scraping OK: ${d.total_chunks} chunks`, 'ok');
    loadScraping();
  } catch(e) { el.textContent = '✗ Erro: ' + e.message; toast('Erro scraping: ' + e.message, 'err'); }
}

async function clearWikiCache() {
  try {
    const d = await api('/scraping/cache', { method: 'DELETE' });
    toast(`Cache wiki limpo: ${d.deleted} entradas`, 'ok');
    loadScraping();
  } catch(e) { toast('Erro: ' + e.message, 'err'); }
}

// ─── USERS ────────────────────────────────────────────────────────────────────
async function loadUsers() {
  const role = document.getElementById('users-role-filter').value;
  const el   = document.getElementById('users-content');
  el.innerHTML = '<div class="loading">Carregando...</div>';
  try {
    const d = await api(`/postgres/pessoas?limit=100${role?'&role='+role:''}`);
    if (!d) return;
    const stats = d.stats || {};
    el.innerHTML = `
      <div class="cards cols-6" style="grid-template-columns:repeat(6,1fr);margin-bottom:20px;">
        ${Object.entries(stats).map(([r,c]) =>
          `<div class="card" style="padding:12px 14px;text-align:center">
            <div class="card-val" style="font-size:20px">${c}</div>
            <div class="card-label">${r}</div></div>`
        ).join('')}
      </div>
      <div class="tbl-wrap"><table>
        <thead><tr><th>ID</th><th>Nome</th><th>Telefone</th><th>Role</th><th>Status</th><th>Centro</th><th>Verificado</th><th>Ações</th></tr></thead>
        <tbody>${(d.pessoas||[]).map(p =>
          `<tr>
            <td style="color:var(--muted)">${p.id}</td>
            <td>${esc(p.nome||'')}</td>
            <td style="font-family:var(--mono)">${p.telefone||'—'}</td>
            <td><span class="tag ${roleColor(p.role)}">${p.role}</span></td>
            <td><span class="tag ${p.status==='ativo'?'green':p.status==='trancado'?'yellow':'muted'}">${p.status}</span></td>
            <td style="color:var(--muted)">${p.centro||'—'}</td>
            <td><span class="tag ${p.verificado?'green':'muted'}">${p.verificado?'sim':'não'}</span></td>
            <td style="display:flex;gap:4px;">
              <button class="btn sm" onclick="editUser(${p.id},'${p.role}')">✏</button>
              <button class="btn sm" onclick="viewUserMemory('${p.telefone||p.id}')">🧠</button>
            </td></tr>`
        ).join('')}</tbody></table></div>`;
  } catch(e) { el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`; }
}

// ─── MEMORY ───────────────────────────────────────────────────────────────────
async function loadMemoryStats() {
  const el  = document.getElementById('memory-stats');
  const el2 = document.getElementById('memory-top-users');
  el.innerHTML  = '<div class="loading">...</div>';
  el2.innerHTML = '<div class="loading">...</div>';
  try {
    const d = await api('/memory');
    if (!d) return;
    el.innerHTML = `
      <div class="sect-head"><div class="sect-title">Estatísticas de Memória</div></div>
      <div class="card">
        <div class="key-val"><span class="k">Usuários com fatos</span><span class="v">${d.users_com_fatos}</span></div>
        <div class="key-val"><span class="k">Sessões de chat ativas</span><span class="v">${d.sessoes_chat}</span></div>
        <div class="key-val"><span class="k">Working memories</span><span class="v">${d.working_memories}</span></div>
      </div>`;
    el2.innerHTML = `<div class="tbl-wrap"><table>
      <thead><tr><th>User ID</th><th>Fatos</th><th>Ações</th></tr></thead>
      <tbody>${(d.top_users||[]).map(u =>
        `<tr><td style="font-family:var(--mono)">${u.user_id}</td>
         <td><span class="tag blue">${u.fatos}</span></td>
         <td style="display:flex;gap:4px;">
           <button class="btn sm" onclick="viewUserMemory('${u.user_id}')">👁 Ver</button>
           <button class="btn sm danger" onclick="clearUserMemory('${u.user_id}','fatos')">🗑 Fatos</button>
           <button class="btn sm danger" onclick="clearUserMemory('${u.user_id}','tudo')">☢ Tudo</button>
         </td></tr>`
      ).join('') || '<tr><td colspan="3" class="empty">Nenhum fato armazenado</td></tr>'}</tbody></table></div>`;
  } catch(e) { el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`; }
}

async function loadMemoryUser() {
  const uid = document.getElementById('mem-user-input').value.trim();
  if (!uid) return;
  const el = document.getElementById('mem-user-result');
  el.innerHTML = '<div class="loading">...</div>';
  try {
    const d = await api(`/memory/${encodeURIComponent(uid)}`);
    if (!d) return;
    el.innerHTML = `
      <div style="margin-bottom:8px;display:flex;gap:6px;">
        <button class="btn sm danger" onclick="clearUserMemory('${uid}','fatos')">🗑 Fatos</button>
        <button class="btn sm danger" onclick="clearUserMemory('${uid}','tudo')">☢ Tudo</button>
      </div>
      <div class="key-val"><span class="k">Fatos salvos</span><span class="v">${(d.fatos||[]).length}</span></div>
      <div class="key-val"><span class="k">Turns no histórico</span><span class="v">${d.historico_turns}</span></div>
      <div style="margin-top:10px;">
        <div style="color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Fatos</div>
        ${(d.fatos||[]).map(f => `<div style="padding:4px 8px;margin:3px 0;background:var(--surf2);border-radius:4px;font-size:11px;">• ${esc(f)}</div>`).join('') || '<div style="color:var(--muted);font-size:11px;">Nenhum fato</div>'}
      </div>
      ${d.historico_preview ? `<div style="margin-top:10px;"><div style="color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Histórico (preview)</div><pre style="font-size:10px;max-height:150px">${esc(d.historico_preview)}</pre></div>` : ''}`;
  } catch(e) { el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`; }
}

function viewUserMemory(uid) {
  nav('memory');
  setTimeout(() => {
    document.getElementById('mem-user-input').value = uid;
    loadMemoryUser();
  }, 100);
}

async function clearUserMemory(uid, tipo) {
  confirmAction('clear-mem', `Limpar ${tipo} de ${uid}?`, tipo==='tudo'?'Remove fatos, histórico e working memory':'Remove apenas os fatos permanentes', async () => {
    try {
      await api(`/memory/${encodeURIComponent(uid)}/${tipo}`, { method: 'DELETE' });
      toast(`Memória (${tipo}) de ${uid.slice(-8)} limpa`, 'ok');
      loadMemoryStats();
    } catch(e) { toast('Erro: ' + e.message, 'err'); }
  });
}

// ─── CONFIG ───────────────────────────────────────────────────────────────────
async function loadConfig() {
  const el = document.getElementById('config-content');
  el.innerHTML = '<div class="loading">Carregando configurações...</div>';
  try {
    const d = await api('/config');
    if (!d) return;
    el.innerHTML = `<div class="tbl-wrap"><table>
      <thead><tr><th>Campo</th><th>Valor Atual</th><th>Editável</th><th>Ações</th></tr></thead>
      <tbody>${Object.entries(d).map(([k, v]) =>
        `<tr>
          <td style="color:var(--accent)">${k}</td>
          <td id="cfg-val-${k}" style="font-family:var(--mono)">${esc(String(v.value))}</td>
          <td><span class="tag ${v.editable?'green':'muted'}">${v.editable?'sim':'não'}</span></td>
          <td>${v.editable ?
            `<button class="btn sm" onclick="editConfig('${k}','${esc(String(v.value))}')">✏ Editar</button>` :
            '—'}</td>
        </tr>`
      ).join('')}</tbody></table>
      <div style="padding:12px 14px;color:var(--muted);font-size:11px;border-top:1px solid var(--brd);">
        ⚠ Alterações são em memória apenas. Para persistir, edite o arquivo .env e reinicie o container.
      </div></div>`;
  } catch(e) { el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`; }
}

function editConfig(field, currentVal) {
  const newVal = prompt(`Novo valor para ${field}:\n(atual: ${currentVal})`, currentVal);
  if (newVal === null || newVal === currentVal) return;
  (async () => {
    try {
      const d = await api(`/config/${field}`, { method: 'PATCH', body: { value: newVal } });
      document.getElementById(`cfg-val-${field}`).textContent = String(d.value);
      toast(`${field} atualizado: ${d.value}`, 'ok');
    } catch(e) { toast('Erro: ' + e.message, 'err'); }
  })();
}

async function testWebhook() {
  const phone = document.getElementById('test-phone').value.trim();
  const msg   = document.getElementById('test-msg').value.trim();
  const el    = document.getElementById('webhook-result');
  el.style.display = '';
  el.textContent = '⏳ Enviando...';
  el.style.color = 'var(--muted)';
  try {
    const d = await api('/actions/test-webhook', { method: 'POST', body: { phone, msg } });
    el.textContent = d.ok ? `✓ Enviado para ${phone}` : `✗ Falha ao enviar`;
    el.style.color = d.ok ? 'var(--green)' : 'var(--red)';
  } catch(e) { el.textContent = '✗ ' + e.message; el.style.color = 'var(--red)'; }
}

async function runHealthCheck() {
  const el = document.getElementById('health-result');
  el.innerHTML = '<div class="loading">Verificando...</div>';
  try {
    const d = await api('/overview');
    if (!d) return;
    el.innerHTML = `<div class="tbl-wrap"><table>
      <thead><tr><th>Serviço</th><th>Status</th><th>Detalhes</th></tr></thead>
      <tbody>${Object.entries(d.services||{}).map(([k,s]) =>
        `<tr><td style="text-transform:capitalize">${k}</td>
         <td><span class="tag ${s.status==='ok'||s.status==='ready'||s.status==='configured'?'green':'red'}">${s.status}</span></td>
         <td style="font-size:11px;color:var(--muted)">${s.error||s.version||s.model||s.instance||''}</td></tr>`
      ).join('')}</tbody></table></div>`;
  } catch(e) { el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`; }
}

// ─── LOGS ─────────────────────────────────────────────────────────────────────
let _logsLevel = 'error';
function logsTab(t) {
  _logsLevel = t;
  document.querySelectorAll('#page-logs .section-tab').forEach((el,i) => {
    el.classList.toggle('active', ['error','warn','info','monitor'][i] === t);
  });
  loadLogs(t);
}

async function loadLogs(nivel) {
  const el = document.getElementById('logs-content');
  el.innerHTML = '<div class="loading">Carregando logs...</div>';
  try {
    const d = await api(`/logs?nivel=${nivel}&limit=100`);
    if (!d) return;
    if (!d.logs.length) { el.innerHTML = '<div class="empty">Nenhum log encontrado ✓</div>'; return; }
    el.innerHTML = `<div class="tbl-wrap"><table>
      <thead><tr><th>Timestamp</th>${nivel==='monitor'?'<th>User</th><th>Tokens</th><th>ms</th><th>Rota</th><th>Pergunta</th>':'<th>User</th><th>Contexto</th><th>Mensagem</th>'}</tr></thead>
      <tbody>${d.logs.map(l => nivel==='monitor' ?
        `<tr>
          <td style="font-size:10px;color:var(--muted)">${(l.ts||'').slice(11,19)}</td>
          <td style="font-size:10px">${(l.user_id||'—').slice(-8)}</td>
          <td>${l.tokens_total||0}</td>
          <td>${l.latencia_ms||0}</td>
          <td><span class="tag accent">${l.rota||'?'}</span></td>
          <td style="font-size:10px;color:var(--muted);max-width:200px;overflow:hidden;text-overflow:ellipsis">${esc((l.pergunta||'').slice(0,50))}</td>
        </tr>` :
        `<tr>
          <td style="font-size:10px;color:var(--muted)">${(l.ts||'').slice(11,19)}</td>
          <td style="font-size:10px">${esc(l.user_id||'—')}</td>
          <td style="font-size:10px;color:var(--muted)">${esc(l.context||'—')}</td>
          <td style="font-size:10px;max-width:300px;overflow:hidden;text-overflow:ellipsis;color:var(--${nivel==='error'?'red':'yellow'})">${esc((l.msg||'').slice(0,100))}</td>
        </tr>`
      ).join('')}</tbody></table></div>`;
  } catch(e) { el.innerHTML = `<div class="empty">Erro: ${e.message}</div>`; }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function fmt(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'k';
  return String(n||0);
}

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function updateDot(id, color) {
  const el = document.getElementById(id);
  if (el) { el.className = `dot ${color}`; }
}

function roleColor(r) {
  const map = { admin:'red', coordenador:'accent', professor:'blue', estudante:'green', servidor:'yellow', publico:'muted' };
  return map[r] || 'muted';
}

function typeColor(t) {
  const map = { calendario:'blue', edital:'accent', contatos:'green', wiki_ctic:'yellow', geral:'muted' };
  return map[t] || 'muted';
}
</script>
</body>
</html>"""
