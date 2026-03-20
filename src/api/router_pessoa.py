from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database import get_db

from src.api.schemas import PessoaCreate, PessoaResponse,PessoaUpdate

from src.application.crud_pessoa import criar_pessoas, buscar_pessoas, buscar_pessoa_por_id, atualizar_pessoa, deletar_pessoa

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

@router.put("/{pessoa_id}",response_model=PessoaResponse)
async def endpoint_atualizar_pessoa (pessoa_id: int, pessoa_in: PessoaUpdate, db :AsyncSession = Depends(get_db)):
    """ 
    Rota para  atualziar dados da pessoa existente
    """
    Pessoa_db = await buscar_pessoa_por_id(session=db, pessoa_id=pessoa_id)
    if not Pessoa_db:
        raise HTTPException(status_code=404, detail = "Pessoa não encontrada")
    
    Pessoa_atualizada = await atualizar_pessoa(session = db, pessoa_db=Pessoa_db, pessoa_in=pessoa_in)
    return Pessoa_atualizada

@router.delete("/{pessoa_id}")
async def endpoint_deletar_pessoa(pessoa_id:int, db: AsyncSession = Depends(get_db)):
    """
    rota para deletar uma pessoa do sistema 
    """
    Pessoa_db = await buscar_pessoa_por_id(session=db, pessoa_id=pessoa_id)
    if not Pessoa_db:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada")
    
    await deletar_pessoa(session=db, pessoa_db=Pessoa_db)
    return{"mensagem":"Pessoa deletada com sucesso!"}

    