"""
api/router_pessoa.py — Rotas da API de Pessoas (v2 — Service Layer)
====================================================================

MUDANÇAS vs v1:
  ANTES: CRUD direto no router (sem Service Layer)
  DEPOIS: Router apenas recebe HTTP → delega para PessoaService

  ANTES: Erros genéricos (404 "not found")
  DEPOIS: Erros descritivos com códigos e dicas (409 "EMAIL_DUPLICADO" + hint)

  ADICIONADO: Endpoint de verificação de acesso para o Oráculo.
  ADICIONADO: Endpoint de listagem de estudantes para notificações Celery.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    PessoaCreate,
    PessoaListResponse,
    PessoaResponse,
    PessoaUpdate,
    SuccessResponse,
)
from src.infrastructure.database import get_db
from src.services.pessoa_service import PessoaService

router = APIRouter(prefix="/pessoas", tags=["Pessoas"])


def get_pessoa_service(db: AsyncSession = Depends(get_db)) -> PessoaService:
    """
    Factory da Dependency Injection.
    FastAPI injeta o 'db' e este helper cria o Service com a sessão correta.
    """
    return PessoaService(db)


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Básico
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=PessoaResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Cadastrar nova pessoa",
    description="""
    Cadastra uma nova pessoa no sistema da UEMA.
    
    **Campos obrigatórios:** `nome`, `email`
    
    **Campos opcionais para estudantes:** `matricula`, `centro`, `curso`, `semestre_ingresso`
    
    **Erros possíveis:**
    - 409: Email/telefone/matrícula já cadastrados
    - 422: Dados inválidos (formato de email, semestre, etc.)
    """,
)
async def criar_pessoa(
    dados: PessoaCreate,
    service: PessoaService = Depends(get_pessoa_service),
) -> PessoaResponse:
    from src.services.pessoa_service import CadastroPessoaDTO
    dto = CadastroPessoaDTO(**dados.model_dump())
    pessoa = await service.cadastrar(dto)
    return PessoaResponse.model_validate(pessoa)


@router.get(
    "/",
    response_model=PessoaListResponse,
    summary="Listar pessoas",
)
async def listar_pessoas(
    limit:   int = Query(default=50, ge=1, le=200, description="Máximo de resultados"),
    offset:  int = Query(default=0, ge=0, description="Posição inicial"),
    service: PessoaService = Depends(get_pessoa_service),
) -> PessoaListResponse:
    pessoas = await service.listar(limit=limit, offset=offset)
    return PessoaListResponse(
        total   = len(pessoas),
        limit   = limit,
        offset  = offset,
        pessoas = [PessoaResponse.model_validate(p) for p in pessoas],
    )


@router.get(
    "/{pessoa_id}",
    response_model=PessoaResponse,
    summary="Buscar pessoa por ID",
)
async def buscar_pessoa(
    pessoa_id: int,
    service:   PessoaService = Depends(get_pessoa_service),
) -> PessoaResponse:
    pessoa = await service.buscar_por_id(pessoa_id)
    return PessoaResponse.model_validate(pessoa)


@router.put(
    "/{pessoa_id}",
    response_model=PessoaResponse,
    summary="Atualizar pessoa",
    description="Atualização parcial — envie apenas os campos a alterar.",
)
async def atualizar_pessoa(
    pessoa_id:   int,
    dados:       PessoaUpdate,
    service:     PessoaService = Depends(get_pessoa_service),
) -> PessoaResponse:
    # model_dump(exclude_unset=True): só os campos enviados pelo cliente
    dados_dict = dados.model_dump(exclude_unset=True)
    pessoa     = await service.atualizar(pessoa_id, dados_dict)
    return PessoaResponse.model_validate(pessoa)


@router.delete(
    "/{pessoa_id}",
    response_model=SuccessResponse,
    summary="Remover pessoa",
)
async def deletar_pessoa(
    pessoa_id: int,
    service:   PessoaService = Depends(get_pessoa_service),
) -> SuccessResponse:
    await service.deletar(pessoa_id)
    return SuccessResponse(message=f"Pessoa {pessoa_id} removida com sucesso.")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints Especiais para o Oráculo
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/acesso/telefone/{telefone}",
    summary="Verificar perfil de acesso pelo WhatsApp",
    description="""
    Endpoint usado internamente pelo Oráculo para verificar o perfil 
    do usuário que enviou uma mensagem.
    
    Retorna as permissões e contexto institucional para personalizar a resposta.
    """,
)
async def verificar_acesso(
    telefone: str,
    service:  PessoaService = Depends(get_pessoa_service),
) -> dict:
    perfil = await service.verificar_acesso_por_telefone(telefone)
    return {
        "nome_display":      perfil.nome_display,
        "role":              perfil.role,
        "esta_ativo":        perfil.esta_ativo,
        "pode_ver_restrito": perfil.pode_ver_restrito,
        "pode_abrir_chamado":perfil.pode_abrir_chamado,
        "eh_admin":          perfil.eh_admin,
        "centro":            perfil.centro,
        "curso":             perfil.curso,
        "cadastrado":        perfil.pessoa_id is not None,
    }


@router.get(
    "/notificacao/estudantes-ativos",
    response_model=list[PessoaResponse],
    summary="Estudantes ativos para notificação",
    description="Lista estudantes com WhatsApp cadastrado — usado pelo Celery para enviar lembretes de prazos.",
)
async def estudantes_para_notificacao(
    service: PessoaService = Depends(get_pessoa_service),
) -> list[PessoaResponse]:
    estudantes = await service.listar_estudantes_para_notificacao()
    return [PessoaResponse.model_validate(e) for e in estudantes]