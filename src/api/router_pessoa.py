from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database import get_db

from src.api.schemas import PessoaCreate, PessoaResponse

from src.application.crud_pessoa import criar_pessoas,buscar_pessoas

router = APIRouter(prefix="/pessoas",tags=["Pessoas"])

@router.post("/", response_model=PessoaResponse)
async def endpoint_criar_pessoa(pessoa:PessoaCreate, db:AsyncSession = Depends(get_db)):
    """
    Rota para criar uma nova pessoa. 
    O FastAPI injeta o 'db' automaticamente graças ao Depends(get_db).
    """
    nova_pessooa = await criar_pessoas(session=db, pessoa_in=pessoa)
    return nova_pessooa

@router.get("/",response_model=list[PessoaResponse])
async def endpoint_listar_pessoas(db: AsyncSession = Depends(get_db) ):
    """
    Rota para listar todas as pessoas cadastradas.
    """
    pessoas = await buscar_pessoas(session=db)
    return pessoas