"""
repositories/pessoa_repository.py — Acesso ao Banco (Repository Pattern)
=========================================================================

O QUE É O REPOSITORY PATTERN E POR QUE USAMOS:
─────────────────────────────────────────────────
  Antes (problema):
    O CRUD estava direto nos endpoints do FastAPI.
    Se mudássemos de PostgreSQL para outro banco, teríamos que reescrever
    cada endpoint. E testar era impossível sem subir um banco real.

  Depois (com Repository):
    PessoaRepository = camada que FALA com o banco (só SQL aqui)
    PessoaService    = camada que PENSA as regras de negócio
    Router/Controller= camada que RECEBE a requisição HTTP

    Para testar PessoaService: substituímos o Repository por um mock.
    Para mudar de Postgres para MySQL: só reescrevemos o Repository.

ANALOGIA:
  Repository = "Almoxarifado" — guarda e recupera dados brutos
  Service    = "Gerente"      — decide o que fazer com os dados
  Router     = "Atendente"    — recebe o pedido e passa para o gerente
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.models import Pessoa, RoleEnum, StatusMatriculaEnum


class PessoaRepository:
    """
    Repositório de acesso a dados de Pessoa.
    
    REGRA IMPORTANTE: Esta classe SÓ faz queries SQL.
    Sem lógica de negócio aqui — isso fica no PessoaService.
    """

    def __init__(self, session: AsyncSession) -> None:
        # A sessão é injetada — nunca criada aqui.
        # Isso garante que a mesma transação é usada em toda a operação.
        self._session = session

    # ── Leitura ───────────────────────────────────────────────────────────────

    async def buscar_por_id(self, pessoa_id: int) -> Pessoa | None:
        """Busca uma pessoa pelo ID primário."""
        result = await self._session.execute(
            select(Pessoa).where(Pessoa.id == pessoa_id)
        )
        return result.scalar_one_or_none()

    async def buscar_por_telefone(self, telefone: str) -> Pessoa | None:
        """
        Busca pelo número de WhatsApp normalizado.
        O DevGuard já normaliza o número (remove @s.whatsapp.net e +).
        Ex: '559889123456' (sem espaços, sem símbolos)
        """
        result = await self._session.execute(
            select(Pessoa).where(Pessoa.telefone == telefone)
        )
        return result.scalar_one_or_none()

    async def buscar_por_email(self, email: str) -> Pessoa | None:
        """Busca por email institucional (case-insensitive)."""
        result = await self._session.execute(
            select(Pessoa).where(Pessoa.email == email.lower().strip())
        )
        return result.scalar_one_or_none()

    async def buscar_por_matricula(self, matricula: str) -> Pessoa | None:
        """Busca por número de matrícula UEMA ou SIAPE."""
        result = await self._session.execute(
            select(Pessoa).where(Pessoa.matricula == matricula.strip())
        )
        return result.scalar_one_or_none()

    async def listar_todos(self, limit: int = 100, offset: int = 0) -> list[Pessoa]:
        """Lista todas as pessoas com paginação."""
        result = await self._session.execute(
            select(Pessoa).limit(limit).offset(offset).order_by(Pessoa.id)
        )
        return list(result.scalars().all())

    async def listar_por_role(self, role: RoleEnum) -> list[Pessoa]:
        """Lista todas as pessoas de um papel específico."""
        result = await self._session.execute(
            select(Pessoa).where(Pessoa.role == role)
        )
        return list(result.scalars().all())

    async def listar_estudantes_ativos(self) -> list[Pessoa]:
        """Lista estudantes com matrícula ativa — usado pelo notificador Celery."""
        result = await self._session.execute(
            select(Pessoa).where(
                Pessoa.role == RoleEnum.estudante,
                Pessoa.status == StatusMatriculaEnum.ativo,
                Pessoa.telefone.isnot(None),  # só quem tem WhatsApp cadastrado
            )
        )
        return list(result.scalars().all())

    # ── Escrita ───────────────────────────────────────────────────────────────

    async def criar(self, dados: dict) -> Pessoa:
        """
        Cria uma nova pessoa no banco.
        
        O commit NÃO é feito aqui — quem decide quando commitar é o Service.
        Isso permite que múltiplas operações façam parte da mesma transação.
        """
        # Normaliza email e telefone antes de salvar
        if "email" in dados:
            dados["email"] = dados["email"].lower().strip()
        if "telefone" in dados:
            dados["telefone"] = _normalizar_telefone(dados["telefone"])

        pessoa = Pessoa(**dados)
        self._session.add(pessoa)
        await self._session.flush()  # persiste sem commit (gera o ID)
        await self._session.refresh(pessoa)  # recarrega do banco (pega defaults)
        return pessoa

    async def atualizar(self, pessoa: Pessoa, dados: dict) -> Pessoa:
        """Atualiza campos de uma pessoa existente."""
        if "email" in dados:
            dados["email"] = dados["email"].lower().strip()
        if "telefone" in dados:
            dados["telefone"] = _normalizar_telefone(dados["telefone"])

        for campo, valor in dados.items():
            setattr(pessoa, campo, valor)

        await self._session.flush()
        await self._session.refresh(pessoa)
        return pessoa

    async def deletar(self, pessoa: Pessoa) -> None:
        """Remove uma pessoa do banco."""
        await self._session.delete(pessoa)
        await self._session.flush()

    # ── Checagens rápidas (evita query completa) ──────────────────────────────

    async def telefone_existe(self, telefone: str) -> bool:
        """Verifica existência sem carregar o objeto completo."""
        result = await self._session.execute(
            select(Pessoa.id).where(Pessoa.telefone == _normalizar_telefone(telefone))
        )
        return result.scalar_one_or_none() is not None

    async def email_existe(self, email: str) -> bool:
        result = await self._session.execute(
            select(Pessoa.id).where(Pessoa.email == email.lower().strip())
        )
        return result.scalar_one_or_none() is not None

    async def matricula_existe(self, matricula: str) -> bool:
        result = await self._session.execute(
            select(Pessoa.id).where(Pessoa.matricula == matricula.strip())
        )
        return result.scalar_one_or_none() is not None


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários internos
# ─────────────────────────────────────────────────────────────────────────────

def _normalizar_telefone(telefone: str) -> str:
    """
    Normaliza número de telefone para formato padrão do sistema.
    Remove tudo que não é dígito.
    Ex: '+55 (98) 98912-3456' → '5598989123456'
    """
    if not telefone:
        return telefone
    import re
    return re.sub(r"\D", "", telefone)