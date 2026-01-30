# Usa uma imagem leve do Python
FROM python:3.11-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia os arquivos de requisitos primeiro (para aproveitar cache do Docker)
COPY requirements.txt .

# Instala as dependências
# O --no-cache-dir ajuda a manter a imagem pequena
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do projeto para dentro do container
COPY . .

# Expõe a porta do Streamlit
EXPOSE 8501

# Comando para rodar a aplicação
# address=0.0.0.0 é obrigatório para o Streamlit ser acessível fora do Docker
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]