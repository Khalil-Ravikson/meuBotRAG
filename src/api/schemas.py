"""
api/schemas.py — Pydantic models de request/response
"""
from __future__ import annotations
from pydantic import BaseModel, EmailStr, ConfigDict
from src.domain.models import RoleEnum
from typing import Any
from typing import Optional

class WahaPayload(BaseModel):
    """Payload bruto do WAHA — validação mínima (o DevGuard faz o resto)."""
    event:   str = ""
    payload: dict[str, Any] = {}

    class Config:
        extra = "allow"


class HealthResponse(BaseModel):
    status:    str
    redis:     bool
    agente:    bool
    dev_mode:  bool
    version:   str = "2.0"


class PessoaBase(BaseModel):
    nome : str
    email: EmailStr
    role : RoleEnum = RoleEnum.estudante
    telefone : Optional[str] = None
class PessoaCreate(PessoaBase):
    pass

class PessoaResponse(PessoaBase):
    id: int
    
    model_config = ConfigDict(from_attributes=True)
    
class PessoaUpdate (BaseModel):
    nome : Optional[str] = None
    email: Optional[EmailStr] = None
    role : Optional[RoleEnum] = None
    telefone : Optional[str] = None
