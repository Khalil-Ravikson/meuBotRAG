# Usa Python 3.10 Slim (Leve e estável)
FROM python:3.10-slim

# Define pasta de trabalho
WORKDIR /app

# 1. Instala dependências do SO necessárias para o Postgres e compilação
# libpq-dev é CRUCIAL para conectar no banco
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Copia e instala requirements (Cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copia o código fonte e os dados
COPY src/ ./src
COPY dados/ ./dados

# 4. Define variáveis de ambiente para o Python não criar arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 5. Comando de inicialização
# Usa o uvicorn apontando para a pasta src.main
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]