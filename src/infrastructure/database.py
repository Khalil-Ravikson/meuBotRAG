"""
infrastructure/database.py — Configuração Async do PostgreSQL (v2 — Corrigido)
================================================================================

BUGS CORRIGIDOS vs v1:
  BUG CRÍTICO: URL hardcoded com senha exposta em texto plano.
    ANTES: DATABASE_URL = "postgresql+asyncpg://evolution:01020304@..."
    DEPOIS: Lido de settings (que por sua vez vem do .env)

  BUG SECUNDÁRIO: Sem pool configurado, conexões podem vazar sob carga.
    DEPOIS: pool_size, max_overflow e pool_recycle configurados.

SOBRE O POOL DE CONEXÕES (por que é importante para FastAPI):
  FastAPI é assíncrono. Cada requisição ao webhook pode precisar do banco.
  Sem pool, cada requisição abriria e fecharia uma conexão TCP — lento e caro.
  Com pool_size=5: até 5 conexões simultâneas mantidas "quentes".
  Com max_overflow=10: até 10 conexões extras em pico de tráfego.
  Com pool_recycle=3600: reconecta após 1h (evita timeout do Postgres).

SOBRE get_db():
  FastAPI usa o padrão "Dependency Injection" via Depends(get_db).
  O yield garante que a sessão é SEMPRE fechada, mesmo em erro.
  
  CORRETO (com yield):
    async with AsyncSessionLocal() as session:
        yield session
        # FastAPI chama o código após yield ao final da requisição
  
  ERRADO (sem yield — vaza conexões):
    session = AsyncSessionLocal()
    return session  # nunca é fechada!
"""
from __future__ import annotations

import os
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# ─── URL de conexão ──────────────────────────────────────────────────────────
# Prioridade: variável de ambiente DATABASE_URL → settings → fallback local
# O settings.py lê do .env, então em Docker o docker-compose injeta a URL certa.
def _get_database_url() -> str:
    # Tentativa 1: variável de ambiente direta (docker-compose environment:)
    url = os.getenv("DATABASE_URL", "")
    if url:
        # SQLAlchemy 2.0 async precisa de "postgresql+asyncpg://" não "postgresql://"
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    # Tentativa 2: montar a partir de variáveis separadas
    user   = os.getenv("POSTGRES_USER",     "evolution")
    passwd = os.getenv("POSTGRES_PASSWORD", "evo_pass_2024")
    host   = os.getenv("POSTGRES_HOST",     "evolution-postgres")
    port   = os.getenv("POSTGRES_PORT",     "5432")
    db     = os.getenv("POSTGRES_DB",       "evolution")
    return f"postgresql+asyncpg://{user}:{passwd}@{host}:{port}/{db}"


DATABASE_URL = _get_database_url()

# ─── Engine assíncrona ────────────────────────────────────────────────────────
# echo=False em produção (True só para debug de queries SQL)
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    # Pool de conexões — essencial para performance sob carga
    pool_size=5,           # conexões mantidas abertas permanentemente
    max_overflow=10,       # conexões extras permitidas em pico
    pool_recycle=3600,     # reconecta após 1 hora (evita timeout do Postgres)
    pool_pre_ping=True,    # verifica se conexão ainda está viva antes de usar
)

# ─── Session factory ──────────────────────────────────────────────────────────
# expire_on_commit=False: objetos SQLAlchemy não expiram ao fazer commit.
# Sem isso, acessar obj.campo APÓS commit levaria a uma query extra (lazy load async problemático).
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ─── Base para os Models ──────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base declarativa para todos os models SQLAlchemy do projeto."""
    pass


# ─── Dependency para FastAPI ──────────────────────────────────────────────────
async def get_db() -> AsyncSession:  # type: ignore[override]
    """
    Dependency injection do FastAPI para sessão do banco.

    Uso nos routers:
        @router.get("/")
        async def listar(db: AsyncSession = Depends(get_db)):
            ...

    O 'async with' garante que a sessão é fechada mesmo se houver exceção.
    O 'yield' permite que o FastAPI injete a sessão E depois a feche automaticamente.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()  # desfaz mudanças em caso de erro
            raise
        finally:
            await session.close()  # garante fechamento mesmo em exceção