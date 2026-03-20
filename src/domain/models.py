import enum
from sqlalchemy import Column, Integer, String , Enum as SQLEnum
from src.infrastructure.database import Base

#Hierarquia
class RoleEnum (str, enum.Enum):
    estudante = "estudante"
    professor = "professor"
    coordenador = "coordenador"
    admin = "admin"

class Pessoa (Base):
    __tablename__ = "Pessoas"
    
    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    
    telefone = Column(String, unique=True, index=True, nullable=True)
    # Aqui aplicamos a hierarquia diretamente na coluna do banco
    role = Column(SQLEnum(RoleEnum),default=RoleEnum.estudante,nullable=False)