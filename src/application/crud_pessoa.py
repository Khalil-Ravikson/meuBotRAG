from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.domain.models import Pessoa
from src.api.schemas import PessoaCreate

async def criar_pessoas (session:AsyncSession, pessoa_in:PessoaCreate ) -> Pessoa:
    nova_pessoa =Pessoa(
        nome=pessoa_in.nome,
        email=pessoa_in.email,
        role=pessoa_in.role
        
    ) 
    session.add(nova_pessoa)
    
    
    await session.commit()
    
    await session.refresh(nova_pessoa)
    
    return nova_pessoa

async def buscar_pessoas(session:AsyncSession) -> list[Pessoa]:
    query = select(Pessoa)
    
    resultado = await session.execute(query)
    
    return resultado.scalars().all()