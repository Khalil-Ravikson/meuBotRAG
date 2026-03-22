"""
application/tasks_notificacao.py — Notificações Proativas de Prazos (Celery Beat)
==================================================================================

O QUE FAZ:
───────────
  Toda manhã às 08:00 (horário de Brasília), o Celery Beat dispara esta task.
  Ela verifica quais eventos do calendário acadêmico estão próximos e
  notifica os alunos cadastrados via WhatsApp.

FLUXO COMPLETO:
───────────────
  08:00 Celery Beat → verificar_e_notificar_prazos()
    │
    ├── calendar_parser.buscar_eventos_para_notificar_hoje()
    │     → Redis (BM25 por datas próximas) → lista de EventoCalendario
    │
    ├── PessoaService.listar_estudantes_para_notificacao()
    │     → PostgreSQL → lista de Pessoa (apenas ativos com WhatsApp)
    │
    ├── Para cada (estudante × evento):
    │     ├── Redis: verifica se já foi notificado hoje (anti-spam)
    │     ├── EvolutionService.enviar_mensagem() → WhatsApp
    │     └── Redis: marca como notificado (TTL 20h)
    │
    └── Registra métricas no Redis para o dashboard

PROTEÇÃO ANTI-SPAM:
────────────────────
  Cada notificação gera uma chave Redis:
    notif:{user_id}:{evento_nome_normalizado}:{data_evento}
  TTL: 20 horas
  
  Se a task rodar duas vezes no mesmo dia (reinício do container),
  a segunda execução vê as chaves e não reenvia.

EXEMPLOS DE MENSAGEM:
─────────────────────
  T-3 dias, matrícula urgente:
    "Olá, João! Lembrete do Oráculo UEMA 🎓
     ⏳ Faltam 3 dias!
     ⚠️ Matrícula de veteranos
     📅 Data: 03/02/2026 a 07/02/2026 (semestre 2026.1)
     Precisa de mais informações? É só me perguntar!"

  T-0, último dia de trancamento:
    "Olá, Maria! Lembrete do Oráculo UEMA 🎓
     🚨 Hoje é o último dia!
     ⚠️ Prazo de Trancamento de Matrícula
     📅 Data: 27/02/2026 (semestre 2026.1)
     Precisa de mais informações? É só me pergundar!"
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from datetime import date, datetime, timezone

from src.infrastructure.celery_app import celery_app
from src.infrastructure.redis_client import get_redis_text
from src.rag.calendar_parser import EventoCalendario, buscar_eventos_para_notificar_hoje

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

_TTL_NOTIFICACAO  = 20 * 3600   # 20 horas — previne reenvio no mesmo dia
_PREFIX_NOTIF     = "notif:"    # chave Redis: notif:{user_id}:{evento}:{data}
_MAX_NOTIF_LOTE   = 50          # máximo de alunos por lote (evita sobrecarga)
_DELAY_ENTRE_MSGS = 1.2         # segundos entre mensagens (respeita limites da API)


# ─────────────────────────────────────────────────────────────────────────────
# Task principal — chamada pelo Celery Beat
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name    = "verificar_e_notificar_prazos",
    bind    = True,
    max_retries = 2,
    default_retry_delay = 300,   # retry em 5min se falhar
)
def verificar_e_notificar_prazos(self) -> dict:
    """
    Task principal do notificador. Chamada diariamente pelo Celery Beat.

    Retorna um dict com métricas da execução para o dashboard.
    """
    inicio = time.monotonic()
    logger.info(
        "🔔 Iniciando verificação de prazos | %s",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    try:
        resultado = asyncio.run(_executar_notificacoes())
    except Exception as exc:
        logger.exception("❌ Falha no verificar_e_notificar_prazos: %s", exc)
        raise self.retry(exc=exc)

    duracao = int((time.monotonic() - inicio) * 1000)
    resultado["duracao_ms"] = duracao
    _registrar_metricas(resultado)

    logger.info(
        "✅ Notificações concluídas | eventos=%d | enviadas=%d | "
        "puladas=%d | erros=%d | %dms",
        resultado["eventos_encontrados"],
        resultado["notificacoes_enviadas"],
        resultado["notificacoes_puladas"],
        resultado["erros"],
        duracao,
    )
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# Task disparável manualmente (admin via WhatsApp "!notificar")
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(name="notificar_evento_especifico", bind=True, max_retries=2)
def notificar_evento_especifico(
    self,
    chat_id_admin:  str,
    dias_frente:    int = 7,
    forcar_reenvio: bool = False,
) -> dict:
    """
    Disparada manualmente pelo admin via WhatsApp: "!notificar 7"
    Permite pré-visualizar ou forçar envio de notificações.
    
    Parâmetros:
      chat_id_admin:  WhatsApp do admin que pediu (para confirmar execução)
      dias_frente:    quantos dias à frente verificar
      forcar_reenvio: se True, ignora cache anti-spam
    """
    try:
        resultado = asyncio.run(
            _executar_notificacoes(
                dias_frente    = dias_frente,
                forcar_reenvio = forcar_reenvio,
            )
        )

        # Confirma resultado para o admin
        msg = (
            f"✅ *Notificações enviadas!*\n\n"
            f"📅 Eventos encontrados: {resultado['eventos_encontrados']}\n"
            f"📤 Mensagens enviadas: {resultado['notificacoes_enviadas']}\n"
            f"⏭️  Puladas (já notificadas): {resultado['notificacoes_puladas']}\n"
            f"❌ Erros: {resultado['erros']}\n\n"
            f"_Próxima execução automática: amanhã às 08:00_"
        )

        asyncio.run(_enviar_whatsapp(chat_id_admin, msg))
        return resultado

    except Exception as exc:
        logger.exception("❌ Falha no notificar_evento_especifico: %s", exc)
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────────────────────────────────────
# Lógica assíncrona principal
# ─────────────────────────────────────────────────────────────────────────────

async def _executar_notificacoes(
    dias_frente:    int  = 7,
    forcar_reenvio: bool = False,
) -> dict:
    """
    Orquestra a busca de eventos + listagem de alunos + envio de mensagens.
    """
    metricas = {
        "data":                   date.today().isoformat(),
        "eventos_encontrados":    0,
        "alunos_consultados":     0,
        "notificacoes_enviadas":  0,
        "notificacoes_puladas":   0,
        "erros":                  0,
        "duracao_ms":             0,
    }

    # ── Passo 1: Busca eventos próximos no Redis ───────────────────────────────
    try:
        if dias_frente == 7:
            # Chamada padrão do Beat — usa filtro "deve notificar hoje"
            eventos = buscar_eventos_para_notificar_hoje()
        else:
            from src.rag.calendar_parser import buscar_eventos_proximos
            eventos = buscar_eventos_proximos(dias_frente=dias_frente)
    except Exception as e:
        logger.error("❌ Falha ao buscar eventos: %s", e)
        metricas["erros"] += 1
        return metricas

    metricas["eventos_encontrados"] = len(eventos)

    if not eventos:
        logger.info("ℹ️  Nenhum evento para notificar hoje.")
        return metricas

    logger.info("📅 Eventos para notificar: %s", [e.nome for e in eventos])

    # ── Passo 2: Busca alunos ativos com WhatsApp no PostgreSQL ───────────────
    try:
        alunos = await _listar_alunos_ativos()
    except Exception as e:
        logger.error("❌ Falha ao listar alunos: %s", e)
        metricas["erros"] += 1
        return metricas

    metricas["alunos_consultados"] = len(alunos)

    if not alunos:
        logger.info("ℹ️  Nenhum aluno ativo com WhatsApp cadastrado.")
        return metricas

    logger.info("👥 Alunos para notificar: %d", len(alunos))

    # ── Passo 3: Envia notificações ───────────────────────────────────────────
    for evento in eventos:
        for aluno in alunos[:_MAX_NOTIF_LOTE]:
            try:
                enviado = await _notificar_aluno(
                    aluno          = aluno,
                    evento         = evento,
                    forcar_reenvio = forcar_reenvio,
                )
                if enviado:
                    metricas["notificacoes_enviadas"] += 1
                else:
                    metricas["notificacoes_puladas"] += 1

                # Pausa entre mensagens para não sobrecarregar a Evolution API
                if enviado:
                    await asyncio.sleep(_DELAY_ENTRE_MSGS)

            except Exception as e:
                logger.error(
                    "❌ Erro ao notificar aluno %s sobre '%s': %s",
                    getattr(aluno, "telefone", "?"), evento.nome, e,
                )
                metricas["erros"] += 1

    return metricas


async def _notificar_aluno(
    aluno,
    evento:         EventoCalendario,
    forcar_reenvio: bool = False,
) -> bool:
    """
    Envia a notificação para um aluno específico.
    Retorna True se enviou, False se pulou (já notificado).
    """
    telefone = getattr(aluno, "telefone", None)
    if not telefone:
        return False

    chat_id = f"{telefone}@s.whatsapp.net"

    # ── Anti-spam: verifica se já foi notificado ──────────────────────────────
    if not forcar_reenvio:
        chave_cache = _chave_notificacao(telefone, evento)
        r = get_redis_text()
        if r.get(chave_cache):
            logger.debug("⏭️  Já notificado: %s sobre '%s'", telefone[:8], evento.nome)
            return False

    # ── Monta mensagem personalizada ──────────────────────────────────────────
    nome_display = (
        getattr(aluno, "nome", "").split()[0]
        if getattr(aluno, "nome", "")
        else ""
    )
    mensagem = evento.mensagem_notificacao(nome_display)

    # ── Envia via WhatsApp ────────────────────────────────────────────────────
    sucesso = await _enviar_whatsapp(chat_id, mensagem)

    if sucesso:
        # Marca como notificado no Redis (anti-spam)
        chave_cache = _chave_notificacao(telefone, evento)
        r = get_redis_text()
        r.setex(chave_cache, _TTL_NOTIFICACAO, "1")
        logger.info(
            "📤 Notificado: %s | evento='%s' | dias_restantes=%d",
            nome_display or telefone[:8],
            evento.nome[:40],
            evento.dias_restantes,
        )

    return sucesso


async def _listar_alunos_ativos():
    """Consulta o PostgreSQL por alunos ativos com WhatsApp."""
    from src.infrastructure.database import AsyncSessionLocal
    from src.services.pessoa_service import PessoaService

    async with AsyncSessionLocal() as session:
        service = PessoaService(session)
        return await service.listar_estudantes_para_notificacao()


async def _enviar_whatsapp(chat_id: str, mensagem: str) -> bool:
    """Envia mensagem via EvolutionService. Retorna True se sucesso."""
    from src.services.evolution_service import EvolutionService

    try:
        svc = EvolutionService()
        return await svc.enviar_mensagem(chat_id, mensagem)
    except Exception as e:
        logger.error("❌ Falha ao enviar WhatsApp para %s: %s", chat_id[:20], e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chave_notificacao(telefone: str, evento: EventoCalendario) -> str:
    """
    Gera chave Redis única para rastrear notificação enviada.
    Formato: notif:{telefone_hash}:{evento_hash}:{data}
    
    Usamos hash para não expor telefones nas chaves Redis.
    """
    tel_hash   = hashlib.md5(telefone.encode()).hexdigest()[:8]
    nome_norm  = re.sub(r"\W+", "_", evento.nome.lower())[:20]
    data_str   = evento.data_inicio.strftime("%Y%m%d")
    return f"{_PREFIX_NOTIF}{tel_hash}:{nome_norm}:{data_str}"


def _registrar_metricas(resultado: dict) -> None:
    """Salva métricas da execução no Redis para o dashboard."""
    try:
        import json
        from datetime import datetime, timezone
        r = get_redis_text()
        entrada = json.dumps({
            **resultado,
            "ts": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False)
        r.lpush("metricas:notificacoes", entrada)
        r.ltrim("metricas:notificacoes", 0, 99)  # guarda últimas 100 execuções
    except Exception as e:
        logger.debug("⚠️  Falha ao registrar métricas: %s", e)