from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.domain.models import Pessoa
from src.api.schemas import PessoaCreate, PessoaUpdate

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

async def buscar_pessoa_por_id(session:AsyncSession, pessoa_id: int) -> Pessoa | None:
    query = select(Pessoa).where(Pessoa.id == pessoa_id)
    resultado = await session.execute(query)
    return resultado.scalar_one_or_none()

async def atualizar_pessoa(session:AsyncSession, pessoa_in: PessoaUpdate, pessoa_db : Pessoa) -> Pessoa:
    dados_novos  = pessoa_in.model_dump(exclude_unset=True)
    
    for campo,valor in dados_novos.items():
        setattr(pessoa_db,campo,valor)
    await session.commit()
    await session.refresh(pessoa_db)
    return pessoa_db 

async def deletar_pessoa(session:AsyncSession,pessoa_db : Pessoa):
    await session.delete(pessoa_db)
    await session.commit()


async def buscar_pessoa_por_telefone(session: AsyncSession, telefone: str) -> Pessoa | None:
    """
    Busca um usuário no banco de dados usando o número de telefone (WhatsApp).
    Retorna o objeto Pessoa se encontrar, ou None se o número não for cadastrado.
    """
    query = select(Pessoa).where(Pessoa.telefone == telefone)
    resultado = await session.execute(query)
    
    # scalar_one_or_none() garante que pegamos apenas 1 resultado ou devolvemos vazio
    return resultado.scalar_one_or_none()    