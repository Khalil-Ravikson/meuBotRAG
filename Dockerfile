# =============================================================================
# Dockerfile — Bot UEMA
# =============================================================================
#
# SOBRE O .env E O DOCKER:
#   O Dockerfile NÃO copia o .env para dentro da imagem.
#   As variáveis chegam ao container em runtime via docker-compose.yml:
#     env_file: .env        → injeta o arquivo inteiro
#     environment: ...      → sobrescreve DATABASE_URL, REDIS_URL, WAHA_BASE_URL
#
# SOBRE O requirements.txt vs pyproject.toml:
#   O Dockerfile usa requirements.txt (mais previsível para builds Docker).
#   O pyproject.toml é para uso local e para configurar pytest — não é
#   "copiado para o Docker" no sentido de ser executado lá.
#
# SOBRE O hot-reload:
#   --reload no CMD + volume ./src:/app/src no docker-compose.yml permitem
#   editar o código e ver a mudança sem reconstruir a imagem.
#   Em produção: remova --reload e o volume ./src do docker-compose.yml.
# =============================================================================

# Python 3.11 slim — estável com pydantic-settings v2 e todas as dependências
# Atualizado de 3.10 para 3.11 (melhor suporte a type hints modernos)
FROM python:3.11-slim

WORKDIR /app

# 1. Dependências do sistema operacional
#    libpq-dev  → psycopg para conectar no PostgreSQL/pgvector
#    gcc, build-essential → compilar extensões C (tiktoken, psutil, etc.)
#    curl → usado no healthcheck do docker-compose
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Instala dependências Python em camada separada
#    Enquanto requirements.txt não mudar, o Docker reutiliza esta camada em cache.
#    Mudar só o código em src/ não refaz o pip install.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copia o código-fonte
#    O volume ./src:/app/src no docker-compose.yml sobrescreve em dev (hot-reload).
COPY src/ ./src/

# 4. Copia os PDFs para ingestão
#    O volume ./dados:/app/dados no docker-compose.yml sobrescreve em dev.
COPY dados/ ./dados/

# 5. Variáveis de ambiente internas do Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 6. Expõe a porta da API
EXPOSE 8000

# 7. Comando de inicialização
#    --reload: hot-reload ativo (remova em produção)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]