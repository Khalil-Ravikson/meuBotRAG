from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = "postgresql+asyncpg://evolution:01020304@evolution-postgres:5432/evolution"

engine = create_async_engine(DATABASE_URL,echo=True)


AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base (DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session