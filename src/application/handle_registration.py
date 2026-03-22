"""
application/handle_registration.py — Handler do Fluxo de Registo Conversacional
==================================================================================

RESPONSABILIDADES:
───────────────────
  1. Detecta números não registados (antes de passarem ao bot principal)
  2. Despacha para o RegistrationService (máquina de estados via Redis)
  3. Quando o registo está COMPLETO, cria o Pessoa no PostgreSQL
  4. Envia mensagens ao utilizador via EvolutionService

INTEGRAÇÃO NO FLUXO (main.py → webhook):
──────────────────────────────────────────
  ANTES (com registo hardcoded em main.py):
    is_valid → verificar DB → não cadastrado → msg genérica → return

  DEPOIS (com este handler):
    is_valid → verificar DB
      → cadastrado:     → celery task → bot normal
      → não cadastrado: → registration_handler → state machine
          → em registo:  → processar próximo passo
          → novo:        → iniciar fluxo
          → completo:    → criar no DB → bot normal

DESIGN DECISIONS:
──────────────────
  1. Handler SÍNCRONO dentro da route (não usa Celery) — o registo precisa
     de latência baixa e o utilizador espera resposta imediata a cada passo.
     (A criação no DB é feita via asyncio dentro do handler async.)

  2. Idempotente — se o mesmo phone chegar duas vezes simultâneas durante
     o registo, o Redis impede race condition (setex atómico).

  3. Role default para emails @aluno.uema.br → "estudante" automático
     (o aluno não precisa de escolher se o domínio já indica o papel).
     Mas a máquina de estados ainda pergunta para permitir override.

  4. Email duplicado — se o email já existe na DB para outro telefone,
     o handler informa o utilizador e sugere contactar suporte.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.infrastructure.redis_client import get_redis_text
from src.services.registration_service import (
    DadosRegisto,
    EstadoRegisto,
    RegistrationService,
)

if TYPE_CHECKING:
    from src.services.evolution_service import EvolutionService

logger = logging.getLogger(__name__)

# Singleton do serviço — inicializado lazy para não falhar no import
_reg_service: RegistrationService | None = None


def get_registration_service() -> RegistrationService:
    global _reg_service
    if _reg_service is None:
        _reg_service = RegistrationService(get_redis_text())
    return _reg_service


# ─────────────────────────────────────────────────────────────────────────────
# Handler principal
# ─────────────────────────────────────────────────────────────────────────────

async def handle_registration(
    chat_id:   str,
    phone:     str,
    body:      str,
    evolution: "EvolutionService",
) -> bool:
    """
    Gere o fluxo completo de registo para um número não cadastrado.

    Retorna:
      True  → o fluxo de registo tratou a mensagem (não passar para o bot)
      False → registo concluído (deve redirecionar para o bot normal)
    
    Fluxo interno:
      1. Verifica se está em registo activo → processa próximo passo
      2. Se não está → inicia novo registo
      3. Se estado COMPLETE → persiste no DB → retorna False (continua para bot)
      4. Se estado ABANDONED → retorna True (mensagem de cancelamento já enviada)
    """
    service = get_registration_service()

    # ── Caso 1: número está em fluxo activo ───────────────────────────────────
    if service.esta_em_registo(phone):
        resultado = service.processar(phone, body)
        await evolution.enviar_mensagem(chat_id, resultado.resposta)

        if resultado.concluido and resultado.dados:
            # Persiste no DB e liberta para o bot normal
            sucesso = await _criar_pessoa_no_db(phone, resultado.dados, chat_id, evolution)
            if sucesso:
                service.limpar(phone)
                return False   # redireciona para bot normal
            # Se falhou a criação no DB, mantém em registo (retry)
            return True

        return True  # continua em registo

    # ── Caso 2: número novo — inicia fluxo ────────────────────────────────────
    logger.info("🆕 Novo número não registado: %s — iniciando registo", phone)
    msg_inicio = service.iniciar(phone)
    await evolution.enviar_mensagem(chat_id, msg_inicio)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Persistência no DB
# ─────────────────────────────────────────────────────────────────────────────

async def _criar_pessoa_no_db(
    phone:     str,
    dados:     DadosRegisto,
    chat_id:   str,
    evolution: "EvolutionService",
) -> bool:
    """
    Cria o registo Pessoa no PostgreSQL.
    
    Trata os casos de erro mais comuns:
      - Email duplicado (já existe noutra conta)
      - Telefone duplicado (número re-registado)
      - Falha de conectividade com a DB
    
    Retorna True se criou com sucesso, False em caso de erro.
    """
    try:
        from src.infrastructure.database import AsyncSessionLocal
        from src.domain.models import Pessoa, RoleEnum, StatusMatriculaEnum

        # Detecta role de forma inteligente com base no domínio do email
        role_str = dados.role or _inferir_role_por_email(dados.email)

        async with AsyncSessionLocal() as session:
            # Verifica se email já existe
            from sqlalchemy.future import select
            result = await session.execute(
                select(Pessoa).where(Pessoa.email == dados.email)
            )
            existente_email = result.scalar_one_or_none()
            if existente_email:
                logger.warning(
                    "⚠️  Email já existe no DB [phone=%s, email=%s]",
                    phone, dados.email,
                )
                await evolution.enviar_mensagem(
                    chat_id,
                    "⚠️ Este e-mail já está associado a outra conta.\n"
                    "Se pensas que é um erro, contacta o suporte:\n"
                    "*ctic@uema.br*",
                )
                return False

            # Verifica se telefone já existe
            result2 = await session.execute(
                select(Pessoa).where(Pessoa.telefone == phone)
            )
            existente_phone = result2.scalar_one_or_none()
            if existente_phone:
                # Telefone já existe mas não estava a aparecer como cadastrado
                # (provavelmente role/status causou exclusão do fluxo normal)
                # Actualiza o registo existente
                logger.info(
                    "ℹ️  Telefone já existe, actualizando [phone=%s]", phone
                )
                existente_phone.nome  = dados.nome
                existente_phone.email = dados.email
                existente_phone.role  = RoleEnum(role_str)
                existente_phone.status= StatusMatriculaEnum.ativo
                await session.commit()
                return True

            nova_pessoa = Pessoa(
                nome     = dados.nome,
                email    = dados.email,
                telefone = phone,
                role     = RoleEnum(role_str),
                status   = StatusMatriculaEnum.ativo,
                verificado= False,   # admin pode verificar manualmente depois
            )
            session.add(nova_pessoa)
            await session.commit()

        logger.info(
            "✅ Pessoa criada no DB [phone=%s, nome=%s, role=%s]",
            phone, dados.nome, role_str,
        )
        return True

    except Exception as e:
        logger.exception("❌ Falha ao criar Pessoa no DB [phone=%s]: %s", phone, e)
        await evolution.enviar_mensagem(
            chat_id,
            "⚠️ Ocorreu um erro técnico ao guardar os teus dados.\n"
            "Tenta novamente ou contacta o suporte: *ctic@uema.br*",
        )
        return False


def _inferir_role_por_email(email: str) -> str:
    """Infere o papel a partir do domínio do email (fallback se role não definido)."""
    email = email.lower()
    if "@aluno.uema.br" in email:
        return "estudante"
    if "@professor.uema.br" in email:
        return "professor"
    if "@servidor.uema.br" in email:
        return "servidor"
    # @uema.br genérico — usa o que o utilizador escolheu ou default
    return "estudante"