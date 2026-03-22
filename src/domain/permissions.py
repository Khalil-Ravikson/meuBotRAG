"""
domain/permissions.py — Sistema de Permissões do Oráculo UEMA
==============================================================

FILOSOFIA DE ACESSO DO ORÁCULO:
─────────────────────────────────
  O Oráculo não é um "porteiro hostil" — ele é um assistente prestativo
  que adapta o nível de ajuda ao contexto do usuário.

  PÚBLICO (visitante):
    Pergunta "Onde fica a UEMA?" → Responde com endereço, campus, horários
    Pergunta "Qual a história?" → Conta a história desde 1972 (FESM)
    Pergunta "Como me inscrever no PAES?" → Explica o processo completo
    Pergunta "Meu histórico escolar" → "Para isso precisa estar cadastrado..."
  
  ESTUDANTE (matriculado e ativo):
    Tudo do público +
    Pode abrir chamados no GLPI
    Recebe contexto do seu curso/centro nas respostas
    Recebe notificações de prazos do seu semestre
    
  SERVIDOR/PROFESSOR:
    Tudo do estudante +
    Pode ver informações administrativas sensíveis
    Acesso a documentos restritos da pró-reitoria
    
  ADMIN/CTIC:
    Tudo +
    Comandos de manutenção (!ingerir, !limpar_cache)
    Dashboard de monitoramento
    Gestão de usuários

LÓGICA DE "DESVIO EDUCADO":
  Quando alguém sem permissão tenta algo restrito, o Oráculo:
  1. Não nega com rispidez
  2. Explica o que é necessário
  3. Oferece uma alternativa ou caminho para se cadastrar
  
  Exemplo:
    Estudante não cadastrado: "Como vejo minhas notas?"
    Oráculo: "Para ver suas notas, você precisa estar cadastrado no sistema. 
               Posso te ajudar com o cadastro agora mesmo! Qual é o seu email 
               institucional UEMA? 😊"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from src.domain.models import RoleEnum, StatusMatriculaEnum


# ─────────────────────────────────────────────────────────────────────────────
# Recursos que o Oráculo pode oferecer
# ─────────────────────────────────────────────────────────────────────────────

class Recurso(str, Enum):
    """
    Recursos/capacidades do Oráculo.
    Cada recurso mapeia para uma ou mais tools do AgentCore.
    """
    # Informação pública — sempre disponível
    INFO_UEMA_GERAL      = "info_uema_geral"       # história, campus, contatos
    INFO_PAES            = "info_paes"             # edital, vagas, inscrições
    INFO_CALENDARIO      = "info_calendario"       # datas, prazos, feriados
    INFO_WIKI_CTIC       = "info_wiki_ctic"        # suporte TI, sistemas

    # Informação institucional — exige cadastro ativo
    HISTORICO_ACADEMICO  = "historico_academico"   # notas, frequência (futuro SIGAA)
    CHAMADO_GLPI         = "chamado_glpi"          # abrir ticket de suporte TI
    NOTIFICACAO_PRAZO    = "notificacao_prazo"     # lembretes de matrícula/prova

    # Administrativo — exige role elevado
    GESTAO_DOCUMENTOS    = "gestao_documentos"     # ingerir/atualizar PDFs
    DASHBOARD_MONITOR    = "dashboard_monitor"     # ver métricas do bot
    GESTAO_USUARIOS      = "gestao_usuarios"       # criar/editar cadastros


# ─────────────────────────────────────────────────────────────────────────────
# Mapeamento de permissões por role
# ─────────────────────────────────────────────────────────────────────────────

_PERMISSOES: dict[RoleEnum, set[Recurso]] = {

    RoleEnum.publico: {
        Recurso.INFO_UEMA_GERAL,
        Recurso.INFO_PAES,
        Recurso.INFO_CALENDARIO,
        Recurso.INFO_WIKI_CTIC,
    },

    RoleEnum.estudante: {
        Recurso.INFO_UEMA_GERAL,
        Recurso.INFO_PAES,
        Recurso.INFO_CALENDARIO,
        Recurso.INFO_WIKI_CTIC,
        Recurso.HISTORICO_ACADEMICO,
        Recurso.CHAMADO_GLPI,
        Recurso.NOTIFICACAO_PRAZO,
    },

    RoleEnum.servidor: {
        Recurso.INFO_UEMA_GERAL,
        Recurso.INFO_PAES,
        Recurso.INFO_CALENDARIO,
        Recurso.INFO_WIKI_CTIC,
        Recurso.CHAMADO_GLPI,
        Recurso.NOTIFICACAO_PRAZO,
    },

    RoleEnum.professor: {
        Recurso.INFO_UEMA_GERAL,
        Recurso.INFO_PAES,
        Recurso.INFO_CALENDARIO,
        Recurso.INFO_WIKI_CTIC,
        Recurso.HISTORICO_ACADEMICO,  # ver histórico dos alunos (futuro)
        Recurso.CHAMADO_GLPI,
        Recurso.NOTIFICACAO_PRAZO,
    },

    RoleEnum.coordenador: {
        Recurso.INFO_UEMA_GERAL,
        Recurso.INFO_PAES,
        Recurso.INFO_CALENDARIO,
        Recurso.INFO_WIKI_CTIC,
        Recurso.HISTORICO_ACADEMICO,
        Recurso.CHAMADO_GLPI,
        Recurso.NOTIFICACAO_PRAZO,
        Recurso.GESTAO_DOCUMENTOS,    # pode ingerir documentos do curso
    },

    RoleEnum.admin: {
        # Admin tem acesso a tudo
        Recurso.INFO_UEMA_GERAL,
        Recurso.INFO_PAES,
        Recurso.INFO_CALENDARIO,
        Recurso.INFO_WIKI_CTIC,
        Recurso.HISTORICO_ACADEMICO,
        Recurso.CHAMADO_GLPI,
        Recurso.NOTIFICACAO_PRAZO,
        Recurso.GESTAO_DOCUMENTOS,
        Recurso.DASHBOARD_MONITOR,
        Recurso.GESTAO_USUARIOS,
    },
}

# Status que bloqueiam acesso a recursos institucionais
_STATUS_BLOQUEADOS = {StatusMatriculaEnum.inativo, StatusMatriculaEnum.pendente}


@dataclass
class ContextoPermissao:
    """
    Contexto de permissão calculado para um usuário específico.
    Usado pelo Oráculo para tomar decisões em tempo real.
    """
    role:               RoleEnum
    status:             StatusMatriculaEnum
    recursos_permitidos: set[Recurso] = field(default_factory=set)
    nome_display:       str = "visitante"
    centro:             str | None = None
    curso:              str | None = None

    def pode(self, recurso: Recurso) -> bool:
        """Verifica se o usuário pode acessar um recurso específico."""
        # Recursos públicos sempre permitidos
        recursos_publicos = _PERMISSOES[RoleEnum.publico]
        if recurso in recursos_publicos:
            return True

        # Para outros recursos: verifica role E status
        if self.status in _STATUS_BLOQUEADOS:
            return False

        return recurso in self.recursos_permitidos

    def lista_tools_permitidas(self) -> list[str]:
        """
        Retorna a lista de tools do AgentCore que este usuário pode usar.
        Usado em handle_message.py para filtrar o que o agente pode chamar.
        """
        mapeamento: dict[Recurso, list[str]] = {
            Recurso.INFO_CALENDARIO:    ["consultar_calendario_academico"],
            Recurso.INFO_PAES:          ["consultar_edital_paes_2026"],
            Recurso.INFO_UEMA_GERAL:    ["consultar_contatos_uema"],
            Recurso.INFO_WIKI_CTIC:     ["consultar_wiki_ctic"],
            Recurso.CHAMADO_GLPI:       ["abrir_chamado_glpi"],
            Recurso.GESTAO_DOCUMENTOS:  ["admin_limpar_cache", "admin_status_sistema"],
        }

        tools = []
        for recurso, tool_list in mapeamento.items():
            if self.pode(recurso):
                tools.extend(tool_list)
        return list(set(tools))  # remove duplicatas

    def mensagem_sem_permissao(self, recurso: Recurso) -> str:
        """
        Gera mensagem educada para quando o usuário tenta algo que não pode.
        O Oráculo usa isto para redirecionar sem ser rude.
        """
        if self.status == StatusMatriculaEnum.pendente:
            return (
                "Para acessar essa informação, você precisa completar seu cadastro. "
                "Me diga seu email institucional UEMA e te ajudo agora mesmo! 📝"
            )
        if self.status == StatusMatriculaEnum.inativo:
            return (
                "Parece que seu vínculo com a UEMA está inativo. "
                "Se acredita ser um erro, entre em contato com a secretaria do seu curso "
                "ou com o CTIC pelo email ctic@uema.br. 😊"
            )
        if self.role == RoleEnum.publico:
            return (
                "Essa informação é exclusiva para a comunidade UEMA (alunos, professores e servidores). "
                "Se você tem vínculo com a universidade, posso te ajudar a se cadastrar! "
                "Qual é o seu email institucional? 🎓"
            )
        return "Você não tem permissão para acessar este recurso."


def calcular_permissoes(
    role: RoleEnum,
    status: StatusMatriculaEnum,
    nome_display: str = "visitante",
    centro: str | None = None,
    curso: str | None = None,
) -> ContextoPermissao:
    """
    Calcula as permissões para um usuário com base em seu role e status.
    
    Chamado pelo handle_message.py após a verificação no banco de dados.
    """
    # Status bloqueado = apenas recursos públicos
    if status in _STATUS_BLOQUEADOS:
        recursos = _PERMISSOES[RoleEnum.publico]
    else:
        recursos = _PERMISSOES.get(role, _PERMISSOES[RoleEnum.publico])

    return ContextoPermissao(
        role                = role,
        status              = status,
        recursos_permitidos = recursos,
        nome_display        = nome_display,
        centro              = centro,
        curso               = curso,
    )