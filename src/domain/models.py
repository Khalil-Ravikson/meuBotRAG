"""
domain/models.py — Models SQLAlchemy da UEMA (v2 — Context UEMA Completo)
===========================================================================

CONTEXTO UEMA (por que estes campos importam para o Oráculo):
──────────────────────────────────────────────────────────────
  A UEMA é uma universidade pública multicampi com:
    - 87 municípios | 20 campi | 29.000+ alunos | 1.377 docentes
    - Centros: CECEN, CESB, CESC, CCSA, CEEA, CCS, CCT...
    - Sistemas: SIGAA (acadêmico), SIE (gestão), GLPI (helpdesk TI)
    - Processo Seletivo: PAES (substitui vestibular)
    - CTIC: Centro de TI, vinculado à PROINFRA

  SISTEMA DE PERMISSÕES (por que separa público de institucional):
    O Oráculo precisa saber QUEM está perguntando para decidir O QUE responder.
    
    PÚBLICO (sem cadastro):
      - "Onde fica a UEMA?" → responde
      - "Qual a história da UEMA?" → responde
      - "Como me inscrevo no PAES?" → responde
    
    ESTUDANTE (com matrícula):
      - "Meu histórico escolar" → responde (precisa matrícula)
      - "Abrir chamado no GLPI" → responde (é aluno ativo)
      - "Nota da minha turma" → responde
    
    SERVIDOR/PROFESSOR (com SIAPE):
      - "Relatório de frequência" → responde
      - "Diário de classe no SIGAA" → responde
    
    ADMIN/CTIC:
      - Comandos de ingestão de documentos
      - Acesso ao dashboard de monitoramento
      - Comandos de manutenção
"""
from __future__ import annotations

import enum
from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SQLEnum,
    Integer, String, func,
)
from src.infrastructure.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# Enums institucionais UEMA
# ─────────────────────────────────────────────────────────────────────────────

class RoleEnum(str, enum.Enum):
    """
    Hierarquia de papéis institucionais.
    
    O Oráculo usa este campo para filtrar o que pode responder:
      publico  → acesso mínimo (visitante sem cadastro)
      estudante→ acesso acadêmico padrão
      servidor → acesso administrativo
      professor→ acesso docente
      coordenador → acesso de gestão de curso
      admin    → acesso total + comandos de manutenção
    """
    publico     = "publico"      # visitante externo — sem vínculo
    estudante   = "estudante"    # aluno matriculado na UEMA
    servidor    = "servidor"     # técnico-administrativo
    professor   = "professor"    # docente efetivo ou substituto
    coordenador = "coordenador"  # diretor de curso/departamento
    admin       = "admin"        # CTIC / TI — acesso total


class CentroEnum(str, enum.Enum):
    """
    Centros acadêmicos da UEMA São Luís.
    Fonte: uema.br/campi-e-centros/
    """
    CECEN  = "CECEN"   # Centro de Ciências Exatas e Naturais
    CESB   = "CESB"    # Centro de Estudos Superiores de Bacabal
    CESC   = "CESC"    # Centro de Estudos Superiores de Caxias
    CCSA   = "CCSA"    # Centro de Ciências Sociais Aplicadas
    CEEA   = "CEEA"    # Centro de Educação, Exatas e Agrárias
    CCS    = "CCS"     # Centro de Ciências da Saúde (criado 2023)
    CCT    = "CCT"     # Centro de Ciências Tecnológicas (Caxias)
    CESBA  = "CESBA"   # Centro de Estudos Superiores de Balsas
    OUTRO  = "OUTRO"   # Outros campi/centros do interior


class StatusMatriculaEnum(str, enum.Enum):
    """Status do vínculo institucional — controla acesso do Oráculo."""
    ativo    = "ativo"     # vínculo ativo → acesso pleno ao papel
    inativo  = "inativo"   # desligado/formado → acesso limitado
    trancado = "trancado"  # matrícula trancada → acesso parcial
    pendente = "pendente"  # cadastro aguardando verificação


# ─────────────────────────────────────────────────────────────────────────────
# Model Principal
# ─────────────────────────────────────────────────────────────────────────────

class Pessoa(Base):
    """
    Representa qualquer pessoa com vínculo ou contato com a UEMA.
    
    CAMPOS ESSENCIAIS PARA O ORÁCULO:
      telefone    → identificador no WhatsApp (formato: 55989XXXXXXX)
      role        → papel institucional → define o que o bot pode responder
      status      → ativo/trancado/inativo → controle de acesso
      centro      → localização acadêmica → personaliza respostas
      curso       → área de conhecimento → contextualiza dúvidas
      matricula   → identificador único UEMA → futuro: consulta SIGAA
    
    FLUXO DE CADASTRO NO ORÁCULO (conversacional via WhatsApp):
      1. Usuário manda mensagem → DevGuard identifica telefone
      2. Bot consulta Pessoa pelo telefone → não encontrado?
      3. Bot inicia cadastro conversacional (Redis salva estado temporário)
      4. Coleta nome → email → matrícula/SIAPE → confirma
      5. Persiste no Postgres → acesso liberado ao papel correto
    """
    __tablename__ = "Pessoas"

    # ── Identificação ─────────────────────────────────────────────────────────
    id    = Column(Integer, primary_key=True, index=True)
    nome  = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, index=True, nullable=False)

    # Número WhatsApp (formato normalizado: somente dígitos, ex: 559889123456)
    # Único porque um número = uma pessoa no sistema
    telefone = Column(String(20), unique=True, index=True, nullable=True)

    # ── Vínculo Institucional UEMA ────────────────────────────────────────────
    # Número de matrícula (estudantes) ou SIAPE (servidores/professores)
    matricula = Column(String(20), unique=True, index=True, nullable=True)
    
    # Centro/unidade acadêmica — localiza o aluno institucionalmente
    centro = Column(SQLEnum(CentroEnum), nullable=True)
    
    # Curso de vínculo (texto livre — UEMA tem 78+ cursos)
    # Ex: "Engenharia Civil", "Ciências da Computação", "Medicina Veterinária"
    curso = Column(String(200), nullable=True)
    
    # Semestre de ingresso — contexto para o agente
    # Ex: "2024.1", "2023.2"
    semestre_ingresso = Column(String(10), nullable=True)

    # ── Papel e Acesso ────────────────────────────────────────────────────────
    role   = Column(SQLEnum(RoleEnum),          default=RoleEnum.publico,  nullable=False)
    status = Column(SQLEnum(StatusMatriculaEnum), default=StatusMatriculaEnum.pendente, nullable=False)
    
    # Nível de acesso ao CTIC/GLPI (herda do role, mas pode ser especificado)
    # True = pode abrir chamados no GLPI
    pode_abrir_chamado = Column(Boolean, default=True, nullable=False)

    # ── Metadados ─────────────────────────────────────────────────────────────
    criado_em    = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    atualizado_em= Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Flag para indicar que o cadastro foi verificado manualmente pelo CTIC
    verificado   = Column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return f"<Pessoa id={self.id} nome={self.nome!r} role={self.role} centro={self.centro}>"

    @property
    def display_name(self) -> str:
        """Nome curto para o Oráculo usar em respostas personalizadas."""
        return self.nome.split()[0] if self.nome else "usuário"

    @property
    def esta_ativo(self) -> bool:
        """Verifica se o vínculo institucional está ativo."""
        return self.status == StatusMatriculaEnum.ativo

    @property
    def pode_ver_conteudo_restrito(self) -> bool:
        """
        Conteúdo restrito = informações que exigem vínculo ativo.
        Ex: histórico escolar, chamados GLPI, diário de classe.
        """
        return (
            self.role in (
                RoleEnum.estudante,
                RoleEnum.servidor,
                RoleEnum.professor,
                RoleEnum.coordenador,
                RoleEnum.admin,
            )
            and self.status in (StatusMatriculaEnum.ativo, StatusMatriculaEnum.trancado)
        )