"""
services/registration_service.py — Máquina de Estados de Registo Conversacional
==================================================================================

FLUXO COMPLETO (5 etapas via WhatsApp):
─────────────────────────────────────────
  Bot: "Olá! Não encontrei o teu número. Para te registar, diz-me o teu
        e-mail institucional (@uema.br ou @aluno.uema.br):"
  
  Aluno: "joao.silva@aluno.uema.br"
  
  Bot: "Ótimo! Agora diz-me o teu nome completo:"
  
  Aluno: "João Pedro Silva Santos"
  
  Bot: "Qual é o teu papel na UEMA? Responde com o número:
        1️⃣  Estudante
        2️⃣  Professor
        3️⃣  Servidor Técnico-Administrativo"
  
  Aluno: "1"
  
  Bot: "Confirmar registo? ✅
        Nome:  João Pedro Silva Santos
        Email: joao.silva@aluno.uema.br
        Papel: Estudante
        Responde SIM para confirmar ou NÃO para cancelar."
  
  Aluno: "sim"
  
  Bot: "✅ Registo concluído! Bem-vindo ao Oráculo UEMA, João!
        Já podes fazer as tuas perguntas. 🎓"

ESTADOS NO REDIS:
──────────────────
  Chave: reg:{phone}
  TTL:   15 minutos (abandono automático se o utilizador não responder)
  Valor: JSON com estado actual + dados recolhidos até agora

  Estados:
    WAITING_EMAIL   → a espera do e-mail institucional
    WAITING_NAME    → a espera do nome completo  
    WAITING_ROLE    → a espera da escolha de papel (1/2/3)
    WAITING_CONFIRM → a espera de confirmação (SIM/NÃO)
    ABANDONED       → cancelado ou expirado

VALIDAÇÕES:
────────────
  Email:  deve terminar em @uema.br ou @aluno.uema.br ou @professor.uema.br
          e ter formato válido (não apenas "@uema.br")
  Nome:   mínimo 3 chars, máximo 200, sem números isolados
  Papel:  deve ser "1", "2" ou "3"
  Confirm: "sim", "s", "yes", "y" → confirma
           "não", "nao", "n", "no", "cancelar" → cancela

ANTI-SPAM:
───────────
  Limite de 3 tentativas inválidas por campo antes de abortar o fluxo.
  Chave: reg:attempts:{phone} — TTL 15 minutos.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

_REG_TTL           = 900    # 15 minutos para completar o registo
_ATTEMPTS_TTL      = 900    # TTL para contador de tentativas
_MAX_ATTEMPTS      = 3      # tentativas inválidas antes de abortar
_KEY_PREFIX        = "reg:"
_ATTEMPTS_PREFIX   = "reg:attempts:"

# Domínios de email aceites
_EMAIL_DOMINIOS = (
    "@uema.br",
    "@aluno.uema.br",
    "@professor.uema.br",
    "@servidor.uema.br",
)

# Mapeamento de escolha numérica → role string
_OPCOES_ROLE: dict[str, str] = {
    "1": "estudante",
    "2": "professor",
    "3": "servidor",
}

_LABELS_ROLE: dict[str, str] = {
    "estudante": "Estudante",
    "professor": "Professor",
    "servidor":  "Servidor Técnico-Administrativo",
}


# ─────────────────────────────────────────────────────────────────────────────
# Estado da máquina
# ─────────────────────────────────────────────────────────────────────────────

class EstadoRegisto(str, Enum):
    WAITING_EMAIL   = "WAITING_EMAIL"
    WAITING_NAME    = "WAITING_NAME"
    WAITING_ROLE    = "WAITING_ROLE"
    WAITING_CONFIRM = "WAITING_CONFIRM"
    COMPLETE        = "COMPLETE"
    ABANDONED       = "ABANDONED"


@dataclass
class DadosRegisto:
    """Dados recolhidos durante o fluxo de registo."""
    estado: str  = EstadoRegisto.WAITING_EMAIL.value
    email:  str  = ""
    nome:   str  = ""
    role:   str  = ""
    phone:  str  = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "DadosRegisto":
        d = json.loads(raw)
        return cls(**d)


@dataclass
class ResultadoRegisto:
    """Resultado de um passo da máquina de estados."""
    resposta:    str           # mensagem a enviar ao utilizador
    concluido:   bool = False  # True → criou Pessoa no DB
    abandonado:  bool = False  # True → fluxo cancelado
    continua:    bool = True   # True → fluxo ainda activo
    dados:       DadosRegisto | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Serviço principal
# ─────────────────────────────────────────────────────────────────────────────

class RegistrationService:
    """
    Máquina de estados para registo conversacional via WhatsApp.
    Thread-safe — todo estado persiste no Redis.
    """

    def __init__(self, redis_client):
        """
        Parâmetros:
          redis_client: instância do cliente Redis com decode_responses=True
        """
        self.r = redis_client

    # ── API pública ───────────────────────────────────────────────────────────

    def esta_em_registo(self, phone: str) -> bool:
        """Verifica se um número está actualmente num fluxo de registo."""
        key = f"{_KEY_PREFIX}{phone}"
        try:
            val = self.r.get(key)
            if not val:
                return False
            dados = DadosRegisto.from_json(val)
            return dados.estado not in (
                EstadoRegisto.COMPLETE.value,
                EstadoRegisto.ABANDONED.value,
            )
        except Exception:
            return False

    def iniciar(self, phone: str) -> str:
        """
        Inicia o fluxo de registo para um número novo.
        Retorna a primeira mensagem a enviar.
        """
        dados = DadosRegisto(estado=EstadoRegisto.WAITING_EMAIL.value, phone=phone)
        self._salvar(phone, dados)
        self._reset_attempts(phone)

        logger.info("🆕 Registo iniciado: %s", phone)
        return (
            "👋 *Bem-vindo ao Oráculo UEMA!*\n\n"
            "Não encontrei o teu número no sistema.\n"
            "Para te registares, preciso de alguns dados. Só demora 1 minuto!\n\n"
            "📧 *Qual é o teu e-mail institucional?*\n"
            "_Exemplos: joao@aluno.uema.br, maria@uema.br_"
        )

    def processar(self, phone: str, mensagem: str) -> ResultadoRegisto:
        """
        Processa uma mensagem dentro do fluxo de registo activo.
        
        Retorna ResultadoRegisto com a resposta e o novo estado.
        """
        dados = self._carregar(phone)
        if not dados:
            # Fluxo não existe — inicia um novo
            return ResultadoRegisto(
                resposta=self.iniciar(phone),
                continua=True,
            )

        estado = EstadoRegisto(dados.estado)
        texto  = mensagem.strip()

        # Comando de saída universal
        if _e_cancelamento(texto):
            return self._abandonar(phone, dados, motivo="cancelado pelo utilizador")

        # Despacha para o handler do estado actual
        handlers = {
            EstadoRegisto.WAITING_EMAIL:   self._handle_email,
            EstadoRegisto.WAITING_NAME:    self._handle_nome,
            EstadoRegisto.WAITING_ROLE:    self._handle_role,
            EstadoRegisto.WAITING_CONFIRM: self._handle_confirmacao,
        }
        handler = handlers.get(estado)
        if handler:
            return handler(phone, dados, texto)

        # Estado inválido → reinicia
        return self._abandonar(phone, dados, motivo="estado inválido")

    # ── Handlers de cada estado ───────────────────────────────────────────────

    def _handle_email(self, phone: str, dados: DadosRegisto, texto: str) -> ResultadoRegisto:
        email = texto.lower().strip()

        if not _validar_email(email):
            tentativas = self._incrementar_attempts(phone)
            if tentativas >= _MAX_ATTEMPTS:
                return self._abandonar(
                    phone, dados,
                    motivo="demasiadas tentativas inválidas de email",
                )
            restantes = _MAX_ATTEMPTS - tentativas
            return ResultadoRegisto(
                resposta=(
                    f"❌ E-mail inválido. O e-mail deve terminar em:\n"
                    f"• @aluno.uema.br\n"
                    f"• @uema.br\n"
                    f"• @professor.uema.br\n\n"
                    f"Tentativa {tentativas}/{_MAX_ATTEMPTS}. "
                    f"({restantes} restante{'s' if restantes > 1 else ''}) "
                    f"Tenta novamente:"
                ),
                continua=True,
            )

        self._reset_attempts(phone)
        dados.email  = email
        dados.estado = EstadoRegisto.WAITING_NAME.value
        self._salvar(phone, dados)

        logger.debug("📧 Email aceite [%s]: %s", phone, email)
        return ResultadoRegisto(
            resposta=(
                f"✅ Email registado: *{email}*\n\n"
                "👤 *Qual é o teu nome completo?*\n"
                "_Ex: Maria da Silva Santos_"
            ),
            continua=True,
            dados=dados,
        )

    def _handle_nome(self, phone: str, dados: DadosRegisto, texto: str) -> ResultadoRegisto:
        nome = " ".join(texto.split())  # normaliza espaços

        if not _validar_nome(nome):
            tentativas = self._incrementar_attempts(phone)
            if tentativas >= _MAX_ATTEMPTS:
                return self._abandonar(phone, dados, motivo="nome inválido repetido")
            return ResultadoRegisto(
                resposta=(
                    f"❌ Nome inválido. Precisamos do teu nome completo "
                    f"(mínimo 2 palavras, sem números).\n"
                    f"Tentativa {tentativas}/{_MAX_ATTEMPTS}. Tenta novamente:"
                ),
                continua=True,
            )

        self._reset_attempts(phone)
        dados.nome   = nome
        dados.estado = EstadoRegisto.WAITING_ROLE.value
        self._salvar(phone, dados)

        logger.debug("👤 Nome aceite [%s]: %s", phone, nome)
        return ResultadoRegisto(
            resposta=(
                f"✅ Nome: *{nome}*\n\n"
                "🏫 *Qual é o teu papel na UEMA?*\n\n"
                "1️⃣  Estudante\n"
                "2️⃣  Professor\n"
                "3️⃣  Servidor Técnico-Administrativo\n\n"
                "_Responde apenas com o número (1, 2 ou 3)_"
            ),
            continua=True,
            dados=dados,
        )

    def _handle_role(self, phone: str, dados: DadosRegisto, texto: str) -> ResultadoRegisto:
        opcao = texto.strip().lower()

        # Tolera variações: "1.", "1 ", "estudante", "aluno"
        if opcao in ("estudante", "aluno", "aluna"):
            opcao = "1"
        elif opcao in ("professor", "professora", "docente"):
            opcao = "2"
        elif opcao in ("servidor", "servidora", "técnico", "tecnico", "administrativo"):
            opcao = "3"

        if opcao not in _OPCOES_ROLE:
            tentativas = self._incrementar_attempts(phone)
            if tentativas >= _MAX_ATTEMPTS:
                return self._abandonar(phone, dados, motivo="opção de papel inválida")
            return ResultadoRegisto(
                resposta=(
                    f"❌ Opção inválida. Responde apenas com *1*, *2* ou *3*.\n"
                    f"Tentativa {tentativas}/{_MAX_ATTEMPTS}."
                ),
                continua=True,
            )

        self._reset_attempts(phone)
        dados.role   = _OPCOES_ROLE[opcao]
        dados.estado = EstadoRegisto.WAITING_CONFIRM.value
        self._salvar(phone, dados)

        label_role = _LABELS_ROLE.get(dados.role, dados.role.title())
        logger.debug("🎓 Papel aceite [%s]: %s", phone, dados.role)

        return ResultadoRegisto(
            resposta=(
                f"📋 *Confirmar registo?*\n\n"
                f"👤 Nome:  {dados.nome}\n"
                f"📧 Email: {dados.email}\n"
                f"🏫 Papel: {label_role}\n\n"
                f"Responde *SIM* para confirmar ou *NÃO* para cancelar."
            ),
            continua=True,
            dados=dados,
        )

    def _handle_confirmacao(self, phone: str, dados: DadosRegisto, texto: str) -> ResultadoRegisto:
        resp = texto.strip().lower()

        if resp in ("sim", "s", "yes", "y", "confirmo", "confirmar", "ok", "1"):
            return self._concluir(phone, dados)

        if resp in ("não", "nao", "n", "no", "cancelar", "cancelo", "0"):
            return self._abandonar(phone, dados, motivo="recusou na confirmação")

        tentativas = self._incrementar_attempts(phone)
        if tentativas >= _MAX_ATTEMPTS:
            return self._abandonar(phone, dados, motivo="confirmação não recebida")

        return ResultadoRegisto(
            resposta=(
                "Responde *SIM* para confirmar o registo "
                "ou *NÃO* para cancelar.\n"
                f"Tentativa {tentativas}/{_MAX_ATTEMPTS}."
            ),
            continua=True,
        )

    # ── Acções finais ─────────────────────────────────────────────────────────

    def _concluir(self, phone: str, dados: DadosRegisto) -> ResultadoRegisto:
        """Marca como COMPLETE — o handler assíncrono criará o Pessoa no DB."""
        dados.estado = EstadoRegisto.COMPLETE.value
        self._salvar(phone, dados)

        primeiro_nome = dados.nome.split()[0]
        label_role    = _LABELS_ROLE.get(dados.role, dados.role.title())

        logger.info(
            "✅ Registo concluído [%s]: nome=%s | email=%s | role=%s",
            phone, dados.nome, dados.email, dados.role,
        )

        return ResultadoRegisto(
            resposta=(
                f"✅ *Registo concluído! Bem-vindo(a), {primeiro_nome}!*\n\n"
                f"O teu perfil foi criado:\n"
                f"• Papel: {label_role}\n"
                f"• Email: {dados.email}\n\n"
                "Agora já podes usar o Oráculo UEMA! 🎓\n"
                "Faz a tua primeira pergunta:"
            ),
            concluido=True,
            continua=False,
            dados=dados,
        )

    def _abandonar(
        self,
        phone: str,
        dados: DadosRegisto,
        motivo: str = "",
    ) -> ResultadoRegisto:
        dados.estado = EstadoRegisto.ABANDONED.value
        self._salvar(phone, dados, ttl=60)  # TTL curto — limpeza rápida
        self._reset_attempts(phone)
        logger.info("🚫 Registo abandonado [%s]: %s", phone, motivo)

        return ResultadoRegisto(
            resposta=(
                "❌ *Registo cancelado.*\n\n"
                "Podes tentar novamente quando quiseres — "
                "basta enviares qualquer mensagem. 👋"
            ),
            abandonado=True,
            continua=False,
            dados=dados,
        )

    # ── Persistência ──────────────────────────────────────────────────────────

    def _salvar(self, phone: str, dados: DadosRegisto, ttl: int = _REG_TTL) -> None:
        try:
            self.r.setex(f"{_KEY_PREFIX}{phone}", ttl, dados.to_json())
        except Exception as e:
            logger.warning("⚠️  Falha ao salvar estado de registo [%s]: %s", phone, e)

    def _carregar(self, phone: str) -> DadosRegisto | None:
        try:
            raw = self.r.get(f"{_KEY_PREFIX}{phone}")
            if raw:
                return DadosRegisto.from_json(raw)
        except Exception as e:
            logger.warning("⚠️  Falha ao carregar estado de registo [%s]: %s", phone, e)
        return None

    def _incrementar_attempts(self, phone: str) -> int:
        key = f"{_ATTEMPTS_PREFIX}{phone}"
        try:
            cnt = self.r.incr(key)
            if cnt == 1:
                self.r.expire(key, _ATTEMPTS_TTL)
            return int(cnt)
        except Exception:
            return 1

    def _reset_attempts(self, phone: str) -> None:
        try:
            self.r.delete(f"{_ATTEMPTS_PREFIX}{phone}")
        except Exception:
            pass

    def get_dados(self, phone: str) -> DadosRegisto | None:
        """Retorna os dados de registo para persistir no DB após COMPLETE."""
        return self._carregar(phone)

    def limpar(self, phone: str) -> None:
        """Remove o estado de registo (após criação no DB ou timeout)."""
        try:
            self.r.delete(f"{_KEY_PREFIX}{phone}")
            self.r.delete(f"{_ATTEMPTS_PREFIX}{phone}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de validação (funções puras — fáceis de testar)
# ─────────────────────────────────────────────────────────────────────────────

def _validar_email(email: str) -> bool:
    """
    Valida e-mail institucional UEMA.
    
    Critérios:
      1. Formato básico de email (user@domain.tld)
      2. Parte local não vazia (mínimo 2 chars)
      3. Domínio na lista de domínios UEMA aceites
    
    >>> _validar_email("joao@aluno.uema.br")   → True
    >>> _validar_email("@aluno.uema.br")        → False
    >>> _validar_email("joao@gmail.com")        → False
    >>> _validar_email("joao.silva@uema.br")    → True
    """
    email = email.lower().strip()
    if not re.match(r'^[a-zA-Z0-9._%+\-]{2,}@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email):
        return False
    return any(email.endswith(dom) for dom in _EMAIL_DOMINIOS)


def _validar_nome(nome: str) -> bool:
    """
    Valida nome completo.
    
    Critérios:
      1. Mínimo 2 palavras (nome + apelido)
      2. Mínimo 3 caracteres por nome
      3. Sem sequências de números isolados
      4. Máximo 200 chars total
    
    >>> _validar_nome("João Silva")           → True
    >>> _validar_nome("João")                  → False  (1 palavra)
    >>> _validar_nome("Jo 123")               → False  (contém número)
    >>> _validar_nome("A B")                  → False  (muito curto)
    """
    if not nome or len(nome) > 200:
        return False
    palavras = nome.split()
    if len(palavras) < 2:
        return False
    # Cada palavra deve ter pelo menos 2 chars e ser maioritariamente letras
    for p in palavras:
        if len(p) < 2:
            return False
        # Permite nomes com hífen (Maria-João) mas não números puros
        if re.match(r'^\d+$', p):
            return False
    return True


def _e_cancelamento(texto: str) -> bool:
    """Detecta intenção de cancelar o fluxo."""
    return texto.strip().lower() in (
        "cancelar", "cancel", "sair", "exit", "parar", "stop",
        "esquecer", "desistir", "não quero", "nao quero",
        "/cancelar", "/sair", "/stop",
    )