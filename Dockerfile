FROM python:3.11-slim

# Instala o uv dentro do container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copia os arquivos de configuração do uv (em vez do requirements)
COPY pyproject.toml uv.lock ./

# Instala as dependências de forma idêntica ao seu PC
RUN uv sync --frozen --no-cache

# Copia o resto do código
COPY . .

# Comando para rodar (usando o ambiente do uv)
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]