# ⚖️ Juridico AI — Análise de Leis para ME/EPP

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/AI-Bedrock-orange.svg" alt="AWS Bedrock">
  <img src="https://img.shields.io/badge/Database-Qdrant-red.svg" alt="Qdrant">
</p>

Aplicação de consultoria jurídica assistida por IA, focada em **Direito Empresarial para Micro e Pequenas Empresas (ME/EPP)**. O sistema combina **RAG (Retrieval Augmented Generation)** com **agentes especializados** e uma interface intuitiva para chat e ingestão de leis.

---

## 🚀 Visão Geral

- 💬 **Chat Jurídico**: Interface conversacional com classificação automática de intenções.
- 🤖 **Agentes Especialistas**: Agentes de domínio (Tributário, Trabalhista, Societário) com regras de resposta personalizadas.
- 📚 **RAG com Qdrant**: Respostas fundamentadas em legislação atualizada e indexada.
- 📥 **Ingestão Dinâmica**: Indexação de leis diretamente via URLs do Planalto/HTML.
- 🔒 **Segurança**: Autenticação robusta via variáveis de ambiente.

## 🏗️ Arquitetura

O sistema é dividido em componentes modulares para facilitar a manutenção e escalabilidade:

1.  **Frontend (Streamlit)**: Interface de usuário para interação, login e gestão documental.
2.  **Orquestrador (LangGraph)**: Gerencia o fluxo da conversa e roteamento de intenções.
3.  **Agentes (PydanticAI)**: Agentes especializados com prompts direcionados à legislação ME/EPP.
4.  **Base de Conhecimento (Qdrant)**: Database vetorial para busca semântica eficiente.
5.  **Modelos (AWS Bedrock)**: Utiliza Anthropic Claude e modelos de embeddings de alta performance.

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: [Python 3.12+](https://www.python.org/)
- **Interface**: [Streamlit](https://streamlit.io/)
- **Orquestração**: [LangGraph](https://www.langchain.com/langgraph) & [PydanticAI](https://ai.pydantic.dev/)
- **Framework RAG**: [LlamaIndex](https://www.llamaindex.ai/)
- **Banco Vetorial**: [Qdrant](https://qdrant.tech/)
- **Modelos de IA**: [AWS Bedrock](https://aws.amazon.com/bedrock/) (Claude e Embeddings)

---

## 📂 Estrutura do Projeto

```bash
.
├── app.py          # Interface Streamlit e UI
├── main.py         # Grafo de orquestração e lógica de roteamento
├── Agents.py       # Definição dos agentes e suas ferramentas
├── Prompts.py      # Templates de prompts e regras jurídicas
├── ingestion.py    # Pipeline de processamento e indexação de leis
├── utils.py        # Utilitários de parsing de HTML e fatiamento
├── LLM.py          # Configurações de acesso ao AWS Bedrock
└── requirements.txt # Dependências do projeto
```

---

## ⚙️ Configuração e Execução

### 1. Pré-requisitos

- Python 3.12 ou superior
- Docker (opcional, para execução via container)
- Acesso ao AWS Bedrock (configurado via CLI ou variáveis)

### 2. Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# Qdrant
QDRANT_URL=http://localhost:6333

# Autenticação App
APP_USER=seu_usuario
APP_PASSWORD=sua_senha

# AWS Bedrock
AWS_ACCESS_KEY_ID=sua_key
AWS_SECRET_ACCESS_KEY=seu_secret
AWS_DEFAULT_REGION=us-east-1
```

### 3. Execução Local

```bash
# Instalar dependências
pip install -r requirements.txt

# Iniciar aplicação
streamlit run app.py
```

### 4. Execução via Docker (Recomendado)

```bash
docker-compose up --build
```

Acesse a aplicação em [http://localhost:8501](http://localhost:8501).

---

## 💡 Fluxos de Trabalho

### Fluxo de Chat

1. O usuário submete uma dúvida jurídica.
2. O **Router** analisa a pergunta e a encaminha ao agente especialista correspondente.
3. O agente realiza uma busca **RAG** no Qdrant para encontrar trechos da lei pertinentes.
4. Uma resposta fundamentada é gerada e apresentada ao usuário.

### Fluxo de Ingestão de Leis

1. O usuário fornece URLs de leis (ex: Planalto).
2. O sistema extrai o conteúdo em HTML e faz o fatiamento por artigos (`utils.py`).
3. Os textos são convertidos em vetores e armazenados na coleção do Qdrant.

---

<p align="center">Desenvolvido para facilitar o acesso à informação jurídica em ME/EPP.</p>
