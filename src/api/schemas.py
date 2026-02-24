"""
api/schemas.py — Pydantic models de request/response
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import Any


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