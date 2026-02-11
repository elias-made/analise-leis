FROM python:3.11-slim

# Instala o uv dentro do container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copia apenas os arquivos de dependências primeiro (otimiza o cache do Docker)
COPY pyproject.toml uv.lock ./

# Instala as dependências de forma idêntica ao seu PC
RUN uv sync --frozen --no-cache

# Copia o resto do código da aplicação
COPY . .

# O CMD oficial da imagem. O Docker Compose usará este por padrão.
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]