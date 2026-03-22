"""
services/pessoa_service.py — Lógica de Negócio (Service Layer)
================================================================

O QUE FICA AQUI:
  - Regras de negócio da UEMA
  - Validações que envolvem múltiplas entidades
  - Decisões sobre o que o Oráculo pode responder para cada perfil
  - Gerenciamento de transações (commit/rollback)

O QUE NÃO FICA AQUI:
  - Queries SQL (ficam no Repository)
  - Parsing de requests HTTP (fica no Router)
  - Chamadas à API do WhatsApp (fica no EvolutionService)

EXEMPLO DO FLUXO:
  Router (recebe POST /pessoas) 
    → PessoaService.cadastrar()
      → PessoaRepository.email_existe() [verifica duplicata]
      → PessoaRepository.criar()        [insere no banco]
      → session.commit()                [confirma transação]
    → retorna PessoaResponse
  → FastAPI serializa e envia HTTP 201
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.models import Pessoa, RoleEnum, StatusMatriculaEnum
from src.repositories.pessoa_repository import PessoaRepository

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DTOs (Data Transfer Objects) — o que entra e sai do Service
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CadastroPessoaDTO:
    """Dados necessários para cadastrar uma pessoa."""
    nome:              str
    email:             str
    telefone:          str | None       = None
    matricula:         str | None       = None
    centro:            str | None       = None
    curso:             str | None       = None
    semestre_ingresso: str | None       = None
    role:              RoleEnum         = RoleEnum.publico
    status:            StatusMatriculaEnum = StatusMatriculaEnum.pendente


@dataclass
class PerfilAcessoDTO:
    """
    Resultado da verificação de acesso para o Oráculo.
    
    O Oráculo usa este objeto para decidir:
      - Como se apresentar (Olá, {nome}!)
      - Quais tools liberar (só estudantes podem abrir chamados GLPI)
      - Que tipo de resposta dar (técnica, acadêmica, pública)
    """
    pessoa_id:          int | None
    nome_display:       str           # primeiro nome para saudação
    role:               RoleEnum
    status:             StatusMatriculaEnum
    esta_ativo:         bool
    pode_ver_restrito:  bool          # pode ver conteúdo institucional?
    pode_abrir_chamado: bool          # pode abrir ticket GLPI?
    eh_admin:           bool          # acesso total + comandos de manutenção?
    centro:             str | None    # contexto acadêmico para respostas
    curso:              str | None    # personaliza respostas do RAG


# Perfil padrão para usuários não cadastrados
PERFIL_PUBLICO = PerfilAcessoDTO(
    pessoa_id=None,
    nome_display="visitante",
    role=RoleEnum.publico,
    status=StatusMatriculaEnum.pendente,
    esta_ativo=False,
    pode_ver_restrito=False,
    pode_abrir_chamado=False,
    eh_admin=False,
    centro=None,
    curso=None,
)


class PessoaService:
    """
    Service responsável pelo gerenciamento de pessoas na UEMA.
    
    Criação: PessoaService(session) — a sessão é injetada pelo FastAPI via Depends.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo    = PessoaRepository(session)

    # ─────────────────────────────────────────────────────────────────────────
    # Verificação de Acesso — usado pelo Oráculo a cada mensagem recebida
    # ─────────────────────────────────────────────────────────────────────────

    async def verificar_acesso_por_telefone(self, telefone: str) -> PerfilAcessoDTO:
        """
        Ponto de entrada principal do webhook.
        
        Chamado ANTES de qualquer processamento da mensagem para:
          1. Descobrir quem está falando
          2. Definir o que pode ser respondido
          3. Personalizar a saudação do Oráculo
        
        Retorna PERFIL_PUBLICO se o número não está cadastrado
        → O Oráculo ainda responde, mas com acesso limitado.
        """
        try:
            pessoa = await self._repo.buscar_por_telefone(telefone)
        except Exception as e:
            logger.error("Erro ao verificar acesso para %s: %s", telefone, e)
            return PERFIL_PUBLICO  # falha segura — acesso público

        if not pessoa:
            return PERFIL_PUBLICO

        return _pessoa_para_perfil(pessoa)

    async def verificar_acesso_por_id(self, pessoa_id: int) -> PerfilAcessoDTO:
        """Verifica acesso por ID — usado pelos endpoints REST de admin."""
        pessoa = await self._repo.buscar_por_id(pessoa_id)
        if not pessoa:
            return PERFIL_PUBLICO
        return _pessoa_para_perfil(pessoa)

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD com regras de negócio UEMA
    # ─────────────────────────────────────────────────────────────────────────

    async def cadastrar(self, dados: CadastroPessoaDTO) -> Pessoa:
        """
        Cadastra uma nova pessoa com validações institucionais.
        
        Regras de negócio UEMA:
          - Email deve ser único no sistema
          - Matrícula deve ser única se fornecida
          - Telefone deve ser único se fornecido
          - Role 'admin' só pode ser atribuído por outro admin (implementar depois)
        
        Raises:
          HTTPException 409: Email, telefone ou matrícula já cadastrados
          HTTPException 422: Dados inválidos (email malformado, etc.)
        """
        # Verifica duplicatas ANTES de tentar inserir (evita erro de constraint)
        if await self._repo.email_existe(dados.email):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code":    "EMAIL_DUPLICADO",
                    "message": f"O email '{dados.email}' já está cadastrado no sistema.",
                    "hint":    "Tente recuperar sua senha ou entre em contato com o CTIC.",
                },
            )

        if dados.telefone and await self._repo.telefone_existe(dados.telefone):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code":    "TELEFONE_DUPLICADO",
                    "message": "Este número de WhatsApp já está cadastrado.",
                    "hint":    "Se você trocou de número, acesse pelo email cadastrado.",
                },
            )

        if dados.matricula and await self._repo.matricula_existe(dados.matricula):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code":    "MATRICULA_DUPLICADA",
                    "message": f"A matrícula '{dados.matricula}' já está registrada.",
                    "hint":    "Entre em contato com o CTIC se acreditar ser um erro.",
                },
            )

        # Cria no banco
        pessoa_dict = {
            "nome":              dados.nome,
            "email":             dados.email,
            "telefone":          dados.telefone,
            "matricula":         dados.matricula,
            "centro":            dados.centro,
            "curso":             dados.curso,
            "semestre_ingresso": dados.semestre_ingresso,
            "role":              dados.role,
            "status":            dados.status,
        }
        pessoa = await self._repo.criar({k: v for k, v in pessoa_dict.items() if v is not None})
        await self._session.commit()

        logger.info(
            "✅ Nova pessoa cadastrada: id=%d nome=%s role=%s centro=%s",
            pessoa.id, pessoa.nome, pessoa.role, pessoa.centro,
        )
        return pessoa

    async def atualizar(self, pessoa_id: int, dados: dict) -> Pessoa:
        """
        Atualiza dados de uma pessoa existente.
        
        Raises:
          HTTPException 404: Pessoa não encontrada
          HTTPException 409: Email/telefone duplicado
        """
        pessoa = await self._repo.buscar_por_id(pessoa_id)
        if not pessoa:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code":    "PESSOA_NAO_ENCONTRADA",
                    "message": f"Nenhuma pessoa encontrada com id={pessoa_id}.",
                },
            )

        # Verifica conflitos antes de atualizar
        if "email" in dados and dados["email"] != pessoa.email:
            if await self._repo.email_existe(dados["email"]):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={"code": "EMAIL_DUPLICADO", "message": "Email já em uso."},
                )

        if "telefone" in dados and dados["telefone"] != pessoa.telefone:
            if await self._repo.telefone_existe(dados["telefone"]):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={"code": "TELEFONE_DUPLICADO", "message": "Telefone já em uso."},
                )

        pessoa = await self._repo.atualizar(pessoa, dados)
        await self._session.commit()
        return pessoa

    async def deletar(self, pessoa_id: int) -> None:
        """Remove uma pessoa do sistema."""
        pessoa = await self._repo.buscar_por_id(pessoa_id)
        if not pessoa:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "PESSOA_NAO_ENCONTRADA", "message": f"id={pessoa_id} não encontrado."},
            )
        await self._repo.deletar(pessoa)
        await self._session.commit()

    async def listar(self, limit: int = 100, offset: int = 0) -> list[Pessoa]:
        """Lista pessoas com paginação."""
        return await self._repo.listar_todos(limit=limit, offset=offset)

    async def buscar_por_id(self, pessoa_id: int) -> Pessoa:
        """Busca por ID com 404 automático."""
        pessoa = await self._repo.buscar_por_id(pessoa_id)
        if not pessoa:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "PESSOA_NAO_ENCONTRADA", "message": f"id={pessoa_id} não encontrado."},
            )
        return pessoa

    async def listar_estudantes_para_notificacao(self) -> list[Pessoa]:
        """
        Lista estudantes ativos com WhatsApp cadastrado.
        Usado pelo Celery para notificações proativas de prazos do calendário.
        """
        return await self._repo.listar_estudantes_ativos()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _pessoa_para_perfil(pessoa: Pessoa) -> PerfilAcessoDTO:
    """Converte um objeto Pessoa em PerfilAcessoDTO para o Oráculo."""
    return PerfilAcessoDTO(
        pessoa_id          = pessoa.id,
        nome_display       = pessoa.display_name,
        role               = pessoa.role,
        status             = pessoa.status,
        esta_ativo         = pessoa.esta_ativo,
        pode_ver_restrito  = pessoa.pode_ver_conteudo_restrito,
        pode_abrir_chamado = pessoa.pode_abrir_chamado and pessoa.pode_ver_conteudo_restrito,
        eh_admin           = pessoa.role == RoleEnum.admin,
        centro             = pessoa.centro.value if pessoa.centro else None,
        curso              = pessoa.curso,
    )