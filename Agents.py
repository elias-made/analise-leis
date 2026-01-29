# agents_definition.py
import os
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.bedrock import BedrockConverseModel
from llama_index.core.base.base_query_engine import BaseQueryEngine

# =======================================================
# 1. CONFIGURAÇÃO DE DEPENDÊNCIAS (Injeção do LlamaIndex)
# =======================================================
@dataclass
class LegalDeps:
    query_engine: BaseQueryEngine
    historico_conversa: str  # Passamos o histórico como string para o prompt

# Modelo Bedrock (Reutilizável)
bedrock_model = BedrockConverseModel(
    model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    inference_config={"temperature": 0}
)

# =======================================================
# 2. AGENTE ROUTER (Classificador)
# Aqui o PydanticAI brilha: Ele força a saída a ser um Enum
# =======================================================
OutputPerfil = Literal['micro', 'media', 'grande', 'geral']

router_agent = Agent(
    model=bedrock_model,
    result_type=OutputPerfil, # <--- Garante tipagem forte na saída
    deps_type=LegalDeps
)

@router_agent.system_prompt
def prompt_router(ctx: RunContext[LegalDeps]) -> str:
    return """
    Você é um classificador jurídico. 
    Analise a última mensagem do histórico e defina o porte da empresa.
    """

# =======================================================
# 3. AGENTE ESPECIALISTA (Microempresa)
# =======================================================
micro_agent = Agent(
    model=bedrock_model,
    deps_type=LegalDeps,
    system_prompt="Você é especialista em Simples Nacional e ME/EPP. Use a tool de busca se necessário."
)

# --- Tool do PydanticAI conectada ao LlamaIndex ---
@micro_agent.tool
def buscar_leis(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    """Busca leis no banco de dados Qdrant."""
    response = ctx.deps.query_engine.query(termo_busca)
    return str(response)

# =======================================================
# 4. AGENTE ESPECIALISTA (Média/Grande)
# =======================================================
# (Pode repetir a lógica ou criar um agente genérico parametrizável)
general_agent = Agent(
    model=bedrock_model,
    deps_type=LegalDeps,
    system_prompt="Você é um consultor jurídico sênior para grandes empresas (S.A.)."
)

@general_agent.tool
def buscar_leis(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    """Busca leis no banco de dados Qdrant."""
    response = ctx.deps.query_engine.query(termo_busca)
    return str(response)