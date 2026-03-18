"""
middleware/security_guard.py — SecurityGuard v1.0 (RBAC + Rate Limit + Guardrails)
=====================================================================================

ARQUITECTURA:
──────────────
  O SecurityGuard é uma camada que se encaixa ENTRE o DevGuard e o handle_message.
  Executa DEPOIS do DevGuard validar o payload da Evolution API.

  Fluxo completo:
    Evolution API → DevGuard (validar payload) → SecurityGuard (RBAC + Rate Limit)
                 → Guardrails (comandos regex) → AgentCore (LLM + RAG)

O QUE ESTE FICHEIRO FAZ:
─────────────────────────

  1. RBAC (Role-Based Access Control):
       GUEST   → utilizadores desconhecidos (só lê RAG)
       STUDENT → alunos autenticados (RAG + abrir chamado GLPI)
       ADMIN   → administradores (todas as tools + comandos admin via ZapZap)

     A lista de admins e students é configurada no .env:
       ADMIN_NUMBERS=5598999990001,5598999990002
       STUDENT_NUMBERS=5598999990003   (opcional — todos os outros são GUEST)

  2. RATE LIMITING (Fixed Window):
       GUEST:   10 msgs/minuto, 50/hora
       STUDENT: 30 msgs/minuto, 200/hora
       ADMIN:   sem limite
     Implementado via Redis INCR + EXPIRE (atómico, sem race condition).

  3. COMANDOS ADMIN VIA WHATSAPP (Guardrails adaptado do "Boteco"):
     Admins podem enviar comandos especiais:
       !ingerir [nome_ficheiro.pdf]  → ingere PDF enviado ou filename
       !limpar_cache                 → invalida Semantic Cache
       !status                       → retorna estado do sistema
       !tools                        → lista tools registadas
       !ragas [user_id]              → exporta log para dataset RAGAS
       !reload                       → reinicia AgentCore
     Implementação inspirada no RegexGreeter do guardrails.py do "Boteco".

  4. INGESTÃO VIA ZAPTAP (ADMIN ONLY):
     Quando um ADMIN envia um ficheiro PDF/CSV/DOCX com caption:
       "!ingerir" OU "ingerir isto" OU "adicionar ao conhecimento"
     O sistema:
       a) Faz download do ficheiro via Evolution API /chat/getBase64FromMediaMessage
       b) Salva em /app/dados/uploads/
       c) Dispara Celery task para ingerir (async, não bloqueia resposta)
       d) Confirma ao admin: "✅ Ficheiro recebido. A ingerir em background..."

  5. OBSERVABILIDADE (tiktoken):
     Cada interação regista no Redis:
       monitor:{user_id}:{date} → {tokens_usados, latencia, nivel, rota}
     Exposto no endpoint /monitor do main.py.

INTEGRAÇÃO:
────────────
  Em handle_message.py:
    from src.middleware.security_guard import security_guard

    resultado = security_guard.verificar(
        user_id=user_id,
        body=body,
        has_media=mensagem.has_media,
        msg_type=mensagem.msg_type,
        msg_key_id=identity.get("msg_key_id"),
    )
    if resultado.bloqueado:
        await evolution.enviar_mensagem(chat_id, resultado.resposta)
        return
    if resultado.resposta_rapida:
        await evolution.enviar_mensagem(chat_id, resultado.resposta_rapida)
    if resultado.acao == "LLM":
        # ... chama AgentCore com resultado.tools_disponiveis
    elif resultado.acao == "INGERIR_DOC":
        # ... dispara task de ingestão
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RBAC — Níveis de Permissão
# ─────────────────────────────────────────────────────────────────────────────

class NivelAcesso(str, Enum):
    GUEST   = "GUEST"    # lê RAG, sem autenticação
    STUDENT = "STUDENT"  # RAG + GLPI
    ADMIN   = "ADMIN"    # tudo + comandos admin


# Tools disponíveis por nível
_TOOLS_POR_NIVEL: dict[NivelAcesso, list[str]] = {
    NivelAcesso.GUEST: [
        "consultar_calendario_academico",
        "consultar_edital_paes_2026",
        "consultar_contatos_uema",
        "consultar_wiki_ctic",
    ],
    NivelAcesso.STUDENT: [
        "consultar_calendario_academico",
        "consultar_edital_paes_2026",
        "consultar_contatos_uema",
        "consultar_wiki_ctic",
        "abrir_chamado_glpi",          # STUDENT pode abrir chamados
    ],
    NivelAcesso.ADMIN: [
        "consultar_calendario_academico",
        "consultar_edital_paes_2026",
        "consultar_contatos_uema",
        "consultar_wiki_ctic",
        "abrir_chamado_glpi",
        "admin_limpar_cache",          # ADMIN only
        "admin_status_sistema",        # ADMIN only
        "admin_listar_fatos",          # ADMIN only
    ],
}

# Rate limits por nível: (msgs/minuto, msgs/hora)
_RATE_LIMITS: dict[NivelAcesso, tuple[int, int]] = {
    NivelAcesso.GUEST:   (10, 50),
    NivelAcesso.STUDENT: (30, 200),
    NivelAcesso.ADMIN:   (9999, 9999),  # sem limite
}


# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResultadoSecurity:
    """Resultado da verificação do SecurityGuard."""
    acao:             str             = "LLM"       # LLM | INGERIR_DOC | CMD_ADMIN | BLOQUEADO
    bloqueado:        bool            = False
    resposta:         str             = ""          # se bloqueado → msg de erro
    resposta_rapida:  str             = ""          # resposta imediata (sem LLM)
    nivel:            NivelAcesso     = NivelAcesso.GUEST
    tools_disponiveis:list[str]       = field(default_factory=list)
    parametro:        str             = ""          # parâmetro do comando admin
    precisa_celery:   bool            = False
    msg_key_id:       str             = ""          # para download de media


# ─────────────────────────────────────────────────────────────────────────────
# SecurityGuard principal
# ─────────────────────────────────────────────────────────────────────────────

class SecurityGuard:
    """
    Camada de segurança hierárquica + comandos admin.

    Inicializado no startup com a lista de admins e students do .env.
    Thread-safe — usa Redis atómico para rate limiting.
    """

    def __init__(self, redis_client, settings):
        self.r        = redis_client
        self._admins  = self._parse_numeros(getattr(settings, "ADMIN_NUMBERS",  ""))
        self._students= self._parse_numeros(getattr(settings, "STUDENT_NUMBERS", ""))

        # Compilar regex dos comandos admin (inspirado no Boteco guardrails.py)
        self._re_ingerir    = re.compile(
            r'^[!/](ingerir|ingere|adicionar?|aprender?|treinar?)\b(.*)?$',
            re.IGNORECASE,
        )
        self._re_cache      = re.compile(r'^[!/](limpar_cache|clear_cache|flush_cache)$', re.IGNORECASE)
        self._re_status     = re.compile(r'^[!/](status|saude|health)$', re.IGNORECASE)
        self._re_tools      = re.compile(r'^[!/](tools|ferramentas|rotas)$', re.IGNORECASE)
        self._re_ragas      = re.compile(r'^[!/](ragas|eval|avalia[rç])(\s+\S+)?$', re.IGNORECASE)
        self._re_reload     = re.compile(r'^[!/](reload|reiniciar|restart)$', re.IGNORECASE)
        self._re_fatos      = re.compile(r'^[!/](fatos|facts)\s*(\S+)?$', re.IGNORECASE)

        # Intenções de ingestão em linguagem natural (sem !)
        self._re_ingerir_nl = re.compile(
            r'(ingerir|ingere|adiciona|aprende|treina|inclui|indexa)\s+(isso|este|este doc|esse|esse doc|o doc|o pdf|o csv)',
            re.IGNORECASE,
        )

        logger.info(
            "🛡️  SecurityGuard | admins=%d | students=%d",
            len(self._admins), len(self._students),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────────

    def verificar(
        self,
        user_id:    str,
        body:       str,
        has_media:  bool = False,
        msg_type:   str  = "conversation",
        msg_key_id: str  = "",
    ) -> ResultadoSecurity:
        """
        Verifica permissão, rate limit e interpreta comandos.

        Parâmetros:
          user_id:    número do WhatsApp (já normalizado pelo DevGuard)
          body:       texto da mensagem
          has_media:  True se a mensagem tem anexo
          msg_type:   tipo da mensagem Evolution API
          msg_key_id: ID da mensagem (para download de media)

        Retorna:
          ResultadoSecurity com acao, nivel, tools_disponiveis, etc.
        """
        nivel = self._resolver_nivel(user_id)
        tools = _TOOLS_POR_NIVEL[nivel].copy()
        msg   = (body or "").strip()

        # ── 1. Rate Limit ─────────────────────────────────────────────────────
        if not self._verificar_rate_limit(user_id, nivel):
            return ResultadoSecurity(
                acao      = "BLOQUEADO",
                bloqueado = True,
                resposta  = "⏳ Muitas mensagens seguidas! Aguarda um momento antes de continuar.",
                nivel     = nivel,
            )

        # ── 2. Comandos Admin ─────────────────────────────────────────────────
        if nivel == NivelAcesso.ADMIN:
            cmd_result = self._processar_comando_admin(msg, has_media, msg_type, msg_key_id)
            if cmd_result:
                cmd_result.nivel             = nivel
                cmd_result.tools_disponiveis = tools
                return cmd_result

        # ── 3. Ingestão via Media (ADMIN com ficheiro) ────────────────────────
        if nivel == NivelAcesso.ADMIN and has_media and msg_type in (
            "documentMessage", "imageMessage",
        ):
            # Admin enviou ficheiro + legenda de ingestão
            if self._re_ingerir_nl.search(msg) or self._re_ingerir.match(msg):
                return ResultadoSecurity(
                    acao            = "INGERIR_DOC",
                    resposta_rapida = "📥 Ficheiro recebido! A ingerir em background... ~2 minutos.",
                    nivel           = nivel,
                    tools_disponiveis= tools,
                    msg_key_id      = msg_key_id,
                    precisa_celery  = True,
                )

        # ── 4. LLM normal ─────────────────────────────────────────────────────
        return ResultadoSecurity(
            acao             = "LLM",
            nivel            = nivel,
            tools_disponiveis= tools,
            parametro        = msg,
        )

    def resolver_nivel_publico(self, user_id: str) -> str:
        """Expõe o nível como string para o /monitor."""
        return self._resolver_nivel(user_id).value

    # ─────────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────────

    def _resolver_nivel(self, user_id: str) -> NivelAcesso:
        num = _normalizar_num(user_id)
        if num in self._admins:
            return NivelAcesso.ADMIN
        if num in self._students:
            return NivelAcesso.STUDENT
        return NivelAcesso.GUEST

    def _verificar_rate_limit(self, user_id: str, nivel: NivelAcesso) -> bool:
        """
        Fixed Window rate limiting via Redis INCR.

        Chaves:
          rl:min:{user_id}  → contador por minuto (TTL 60s)
          rl:hr:{user_id}   → contador por hora   (TTL 3600s)
        """
        if nivel == NivelAcesso.ADMIN:
            return True  # admins sem limite

        lim_min, lim_hr = _RATE_LIMITS[nivel]

        try:
            key_min = f"rl:min:{user_id}"
            key_hr  = f"rl:hr:{user_id}"

            pipe = self.r.pipeline(transaction=True)
            pipe.incr(key_min)
            pipe.incr(key_hr)
            resultados = pipe.execute()

            cnt_min, cnt_hr = resultados[0], resultados[1]

            # Define TTL apenas na primeira vez (INCR de 1)
            if cnt_min == 1:
                self.r.expire(key_min, 60)
            if cnt_hr == 1:
                self.r.expire(key_hr, 3600)

            if cnt_min > lim_min or cnt_hr > lim_hr:
                logger.warning(
                    "🚫 Rate limit [%s] nivel=%s | min=%d/%d | hr=%d/%d",
                    user_id, nivel.value, cnt_min, lim_min, cnt_hr, lim_hr,
                )
                return False
            return True

        except Exception as e:
            logger.warning("⚠️  Rate limit Redis indisponível: %s — permitindo.", e)
            return True

    def _processar_comando_admin(
        self,
        msg:        str,
        has_media:  bool,
        msg_type:   str,
        msg_key_id: str,
    ) -> ResultadoSecurity | None:
        """
        Interpreta comandos especiais de admin.
        Retorna ResultadoSecurity se for comando, None se for mensagem normal.

        Inspirado no RegexGreeter do guardrails.py do "Boteco":
          !ingerir → INGERIR_DOC
          !limpar_cache → CMD_ADMIN:LIMPAR_CACHE
          !status → CMD_ADMIN:STATUS
          etc.
        """

        # !ingerir [nome_ficheiro]
        m = self._re_ingerir.match(msg)
        if m:
            parametro = (m.group(2) or "").strip()
            if has_media:
                return ResultadoSecurity(
                    acao            = "INGERIR_DOC",
                    resposta_rapida = f"📥 Ficheiro detectado! A ingerir em background...\nSerei avisado quando terminar.",
                    parametro       = parametro,
                    msg_key_id      = msg_key_id,
                    precisa_celery  = True,
                )
            elif parametro:
                return ResultadoSecurity(
                    acao            = "INGERIR_FICHEIRO",
                    resposta_rapida = f"📂 A tentar ingerir '{parametro}' da pasta /dados...",
                    parametro       = parametro,
                    precisa_celery  = True,
                )
            else:
                return ResultadoSecurity(
                    acao            = "ERRO",
                    resposta_rapida = "ℹ️  Uso: `!ingerir` (com ficheiro anexado) ou `!ingerir nome.pdf`",
                )

        # !limpar_cache
        if self._re_cache.match(msg):
            return ResultadoSecurity(
                acao            = "CMD_ADMIN",
                resposta_rapida = "🗑️  A limpar o Semantic Cache...",
                parametro       = "LIMPAR_CACHE",
                precisa_celery  = True,
            )

        # !status
        if self._re_status.match(msg):
            return ResultadoSecurity(
                acao      = "CMD_ADMIN",
                parametro = "STATUS",
            )

        # !tools
        if self._re_tools.match(msg):
            return ResultadoSecurity(
                acao      = "CMD_ADMIN",
                parametro = "TOOLS",
            )

        # !ragas [user_id]
        m = self._re_ragas.match(msg)
        if m:
            target = (m.group(2) or "").strip()
            return ResultadoSecurity(
                acao            = "CMD_ADMIN",
                resposta_rapida = f"📊 A exportar logs para dataset RAGAS...",
                parametro       = f"RAGAS:{target}",
                precisa_celery  = True,
            )

        # !fatos [user_id]
        m = self._re_fatos.match(msg)
        if m:
            target = (m.group(2) or "").strip()
            return ResultadoSecurity(
                acao      = "CMD_ADMIN",
                parametro = f"FATOS:{target}",
            )

        # !reload
        if self._re_reload.match(msg):
            return ResultadoSecurity(
                acao            = "CMD_ADMIN",
                resposta_rapida = "🔄 A reiniciar AgentCore...",
                parametro       = "RELOAD",
                precisa_celery  = True,
            )

        return None  # não é comando → continua para LLM

    @staticmethod
    def _parse_numeros(raw: str) -> set[str]:
        """Parseia lista de números do .env separados por vírgula."""
        return {
            _normalizar_num(n)
            for n in raw.split(",")
            if n.strip()
        }


def _normalizar_num(numero: str) -> str:
    """Remove @s.whatsapp.net e não-dígitos."""
    return re.sub(r"\D", "", numero.split("@")[0])