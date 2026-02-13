from typing import List
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from llama_index.core.base.base_query_engine import BaseQueryEngine
from pydantic import BaseModel, Field

from ddgs import DDGS
from datetime import datetime
from typing import Literal

import Prompts
from Rag import buscar_com_cache_semantico
from LLM import (
    sonnet_bedrock_model,
)
from utils import montar_prompt_documento, preparar_resumo_router

# =======================================================
# 1. CONFIGURAÇÃO DE DEPENDÊNCIAS
# =======================================================
@dataclass
class LegalDeps:
    query_engine: BaseQueryEngine
    historico_conversa: List[dict]
    documento_texto: str = ""

# =======================================================
# 2. TOOLS
# =======================================================
def tool_buscar_rag(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    return buscar_com_cache_semantico(ctx.deps.query_engine, termo_busca)

def tool_pesquisa_web(ctx: RunContext[LegalDeps], consulta: str) -> str:
    print(f"🌍 PESQUISA WEB (DDG): {consulta}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(consulta, region='br-pt', max_results=3))
            if not results:
                return "Nenhum resultado encontrado na web."
            formatted_results = []
            for i, r in enumerate(results):
                texto = (
                    f"--- RESULTADO #{i+1} ---\n"
                    f"TÍTULO: {r.get('title')}\n"
                    f"🔗 LINK_OBRIGATORIO: {r.get('href')}\n"
                    f"RESUMO: {r.get('body')}\n"
                )
                formatted_results.append(texto)
            return "\n".join(formatted_results)
    except Exception as e:
        return f"Erro na pesquisa web: {str(e)}"

# =======================================================
# 3. AGENTES
# =======================================================
router_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps)

@router_agent.system_prompt
def prompt_router(ctx: RunContext[LegalDeps]) -> str:
    resumo = preparar_resumo_router(ctx.deps.documento_texto)

    return Prompts.router_tmpl.format(
        historico_conversa=ctx.deps.historico_conversa,
        resumo_documento=resumo,
        result_type=Literal["simples", "corporativo", "trabalhista", "societario", "conversational", "out_of_scope"]
    )

# --- Agente Simples ---
simples_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps, tools=[tool_buscar_rag, tool_pesquisa_web])
@simples_agent.system_prompt
def prompt_simples(ctx: RunContext[LegalDeps]) -> str:
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    prompt_documento = montar_prompt_documento(ctx.deps.documento_texto)

    return Prompts.simples_tmpl.format(
        historico_conversa=ctx.deps.historico_conversa, 
        data_atual=data_hoje,
        texto_documento=prompt_documento
    )

# --- Agente Corporativo ---
corporativo_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps, tools=[tool_buscar_rag, tool_pesquisa_web])
@corporativo_agent.system_prompt
def prompt_corporativo(ctx: RunContext[LegalDeps]) -> str:
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    prompt_documento = montar_prompt_documento(ctx.deps.documento_texto)
    
    return Prompts.corporativo_tmpl.format(
        historico_conversa=ctx.deps.historico_conversa, 
        data_atual=data_hoje,
        texto_documento=prompt_documento
    )

# --- Agente Trabalhista ---
trabalhista_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps, tools=[tool_buscar_rag, tool_pesquisa_web])
@trabalhista_agent.system_prompt
def prompt_trabalhista(ctx: RunContext[LegalDeps]) -> str:
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    prompt_documento = montar_prompt_documento(ctx.deps.documento_texto)

    return Prompts.trabalhista_tmpl.format(
        historico_conversa=ctx.deps.historico_conversa, 
        data_atual=data_hoje,
        texto_documento=prompt_documento
    )

# --- Agente Societário ---
societario_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps, tools=[tool_buscar_rag, tool_pesquisa_web])
@societario_agent.system_prompt
def prompt_societario(ctx: RunContext[LegalDeps]) -> str:
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    prompt_documento = montar_prompt_documento(ctx.deps.documento_texto)

    return Prompts.societario_tmpl.format(
        historico_conversa=ctx.deps.historico_conversa, 
        data_atual=data_hoje,
        texto_documento=prompt_documento
    )

# --- Agente Conversacional ---
conversational_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps)
@conversational_agent.system_prompt
def prompt_conversational(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.conversational_tmpl.format(historico_conversa=ctx.deps.historico_conversa)

class MetricasAuditoria(BaseModel):
    fundamentacao: int = Field(description="Nota 1-5 para citações e veracidade legal")
    utilidade: int = Field(description="Nota 1-5 para clareza e solução do problema")
    protocolo_visual: int = Field(description="Nota 1-5 para uso de negritos e ausência de crases")
    tom_de_voz: int = Field(description="Nota 1-5 para profissionalismo e prudência")

class AvaliacaoJuiz(BaseModel):
    metricas: MetricasAuditoria
    aprovado: bool = Field(description="True se todas as métricas forem >= 4")
    justificativa: str = Field(description="Resumo da avaliação das métricas")
    correcao_necessaria: str = Field(description="O que exatamente deve ser corrigido")

# --- Agente Juiz ---
judge_agent = Agent(
    model=sonnet_bedrock_model, 
    deps_type=LegalDeps,
    output_type=AvaliacaoJuiz
)