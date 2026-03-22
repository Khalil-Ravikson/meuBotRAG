"""
api/schemas.py — Schemas Pydantic de Request/Response (v2 — UEMA Context)
==========================================================================

SCHEMAS são a "interface pública" da API.
Eles validam o que entra (request) e formatam o que sai (response).
São DIFERENTES dos Models SQLAlchemy:
  - Model = estrutura do banco de dados
  - Schema = estrutura da API HTTP

SEPARAÇÃO INTENCIONAL:
  PessoaCreate: o que o cliente envia → não inclui campos internos (id, criado_em)
  PessoaResponse: o que a API retorna → não inclui campos sensíveis (hash de senha futura)
  PessoaUpdate: parcial (Optional) → só os campos que o cliente quer mudar
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator
from src.domain.models import CentroEnum, RoleEnum, StatusMatriculaEnum


# ─────────────────────────────────────────────────────────────────────────────
# Schemas de Pessoa (UEMA)
# ─────────────────────────────────────────────────────────────────────────────

class PessoaBase(BaseModel):
    """Campos comuns a todos os schemas de Pessoa."""
    nome:              str
    email:             EmailStr       # Pydantic valida formato de email automaticamente
    role:              RoleEnum       = RoleEnum.publico
    status:            StatusMatriculaEnum = StatusMatriculaEnum.pendente
    telefone:          Optional[str]  = None
    matricula:         Optional[str]  = None  # matrícula UEMA ou SIAPE
    centro:            Optional[CentroEnum] = None
    curso:             Optional[str]  = None  # "Engenharia Civil", "Medicina Veterinária"
    semestre_ingresso: Optional[str]  = None  # "2024.1"

    @field_validator("nome")
    @classmethod
    def nome_nao_vazio(cls, v: str) -> str:
        """Nome deve ter pelo menos 2 caracteres."""
        if len(v.strip()) < 2:
            raise ValueError("Nome deve ter pelo menos 2 caracteres")
        return v.strip().title()  # capitaliza: "JOÃO SILVA" → "João Silva"

    @field_validator("matricula")
    @classmethod
    def matricula_valida(cls, v: Optional[str]) -> Optional[str]:
        """Remove espaços e valida formato mínimo."""
        if v is None:
            return v
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Matrícula deve ter pelo menos 5 caracteres")
        return v

    @field_validator("semestre_ingresso")
    @classmethod
    def semestre_formato(cls, v: Optional[str]) -> Optional[str]:
        """Valida formato 'YYYY.S' ex: '2024.1' ou '2023.2'."""
        import re
        if v is None:
            return v
        if not re.match(r"^\d{4}\.[12]$", v.strip()):
            raise ValueError("Semestre deve estar no formato 'YYYY.1' ou 'YYYY.2'")
        return v.strip()


class PessoaCreate(PessoaBase):
    """Schema para criação de nova pessoa (POST /pessoas)."""
    # Herda tudo de PessoaBase — aqui não tem id, não tem timestamps
    pass


class PessoaUpdate(BaseModel):
    """
    Schema para atualização parcial (PUT /pessoas/{id}).
    
    Todos os campos são Optional: o cliente só envia o que quer mudar.
    Pydantic trata campos não enviados como "não alterar".
    """
    nome:              Optional[str]  = None
    email:             Optional[EmailStr] = None
    telefone:          Optional[str]  = None
    matricula:         Optional[str]  = None
    centro:            Optional[CentroEnum] = None
    curso:             Optional[str]  = None
    semestre_ingresso: Optional[str]  = None
    role:              Optional[RoleEnum] = None
    status:            Optional[StatusMatriculaEnum] = None
    verificado:        Optional[bool] = None


class PessoaResponse(PessoaBase):
    """
    Schema de resposta da API.
    Inclui campos gerados pelo banco (id, timestamps).
    Não inclui campos sensíveis (se houver senha no futuro, fica fora).
    
    ConfigDict(from_attributes=True): permite criar a partir de objetos SQLAlchemy.
    Sem isso, Pydantic não consegue ler objetos do banco diretamente.
    """
    id:            int
    criado_em:     datetime
    verificado:    bool = False
    pode_abrir_chamado: bool = True

    model_config = ConfigDict(from_attributes=True)


class PessoaListResponse(BaseModel):
    """Response de listagem com metadados de paginação."""
    total:   int
    limit:   int
    offset:  int
    pessoas: list[PessoaResponse]


# ─────────────────────────────────────────────────────────────────────────────
# Schemas do Webhook (WhatsApp / Evolution API)
# ─────────────────────────────────────────────────────────────────────────────

class WahaPayload(BaseModel):
    """Payload bruto do webhook — validação mínima (o DevGuard faz o resto)."""
    event:   str = ""
    payload: dict[str, Any] = {}

    model_config = ConfigDict(extra="allow")


class HealthResponse(BaseModel):
    """Response do endpoint /health."""
    status:    str
    redis:     bool
    agente:    bool
    postgres:  bool = False
    dev_mode:  bool
    version:   str = "5.0"
    nome_bot:  str = "Oráculo UEMA"


# ─────────────────────────────────────────────────────────────────────────────
# Schemas de Erro Padronizados
# ─────────────────────────────────────────────────────────────────────────────

class ErroResponse(BaseModel):
    """Formato padrão de erro da API — facilita debug no frontend."""
    code:    str      # "EMAIL_DUPLICADO", "PESSOA_NAO_ENCONTRADA", etc.
    message: str      # mensagem human-readable
    hint:    Optional[str] = None  # sugestão de como resolver


class SuccessResponse(BaseModel):
    """Resposta de sucesso simples para operações sem retorno de dados."""
    success: bool = True
    message: str