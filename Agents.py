import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel
from llama_index.core.base.base_query_engine import BaseQueryEngine

import Prompts
from LLM import (
    sonnet_bedrock_model
)

# =======================================================
# 2. CONFIGURAÇÃO DE DEPENDÊNCIAS
# =======================================================
@dataclass
class LegalDeps:
    query_engine: BaseQueryEngine
    historico_conversa: str

# =======================================================
# 4. AGENTE ROUTER (Classificador)
# =======================================================
router_agent = Agent(
    model=sonnet_bedrock_model,
    deps_type=LegalDeps
)

@router_agent.system_prompt
def prompt_router(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.router_tmpl.format(historico_conversa=ctx.deps.historico_conversa)

# =======================================================
# 5. AGENTE TRIBUTÁRIO (Simples Nacional / LC 123)
# =======================================================
tributario_agent = Agent(
    model=sonnet_bedrock_model,
    deps_type=LegalDeps
)

@tributario_agent.system_prompt
def prompt_tributario(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.tributario_tmpl.format(historico_conversa=ctx.deps.historico_conversa)

@tributario_agent.tool
def tool_buscar_tributario(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    """Busca informações sobre impostos, Simples Nacional, ME/EPP e Pronampe."""
    response = ctx.deps.query_engine.query(termo_busca)
    return str(response)

# =======================================================
# 6. AGENTE TRABALHISTA (CLT)
# =======================================================
trabalhista_agent = Agent(
    model=sonnet_bedrock_model,
    deps_type=LegalDeps
)

@trabalhista_agent.system_prompt
def prompt_trabalhista(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.trabalhista_tmpl.format(historico_conversa=ctx.deps.historico_conversa)

@trabalhista_agent.tool
def tool_buscar_trabalhista(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    """Busca informações sobre leis trabalhistas, CLT, funcionários e demissão."""
    response = ctx.deps.query_engine.query(termo_busca)
    return str(response)

# =======================================================
# 7. AGENTE SOCIETÁRIO (Burocracia / Lei 14.195)
# =======================================================
societario_agent = Agent(
    model=sonnet_bedrock_model,
    deps_type=LegalDeps
)

@societario_agent.system_prompt
def prompt_societario(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.societario_tmpl.format(historico_conversa=ctx.deps.historico_conversa)

@societario_agent.tool
def tool_buscar_societario(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    """Busca informações sobre abertura de empresas, sócios, Lei 14.195 e burocracia."""
    response = ctx.deps.query_engine.query(termo_busca)
    return str(response)


# =======================================================
# 8. AGENTE CONVERSACIONAL (Novo)
# =======================================================
conversational_agent = Agent(
    model=sonnet_bedrock_model,
    deps_type=LegalDeps
)

@conversational_agent.system_prompt
def prompt_conversational(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.conversational_tmpl.format(historico_conversa=ctx.deps.historico_conversa)