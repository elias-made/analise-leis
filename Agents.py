import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel
from llama_index.core.base.base_query_engine import BaseQueryEngine

# =======================================================
# 1. CONFIGURAÇÃO DE AMBIENTE E MODELO
# =======================================================
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

bedrock_model = BedrockConverseModel(
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
)

# =======================================================
# 2. CONFIGURAÇÃO DE DEPENDÊNCIAS
# =======================================================
@dataclass
class LegalDeps:
    query_engine: BaseQueryEngine
    historico_conversa: str

# =======================================================
# 3. AGENTE ROUTER (Classificador)
# =======================================================
router_agent = Agent(
    model=bedrock_model,
    deps_type=LegalDeps
)

@router_agent.system_prompt
def prompt_router(ctx: RunContext[LegalDeps]) -> str:
    return f"""
    Você é um classificador jurídico.
    Analise o histórico e a última mensagem para definir o porte da empresa.
    
    RESPONDA APENAS COM UMA DAS SEGUINTES PALAVRAS (sem pontuação):
    micro
    media
    grande
    geral
    
    Histórico da Conversa:
    {ctx.deps.historico_conversa}
    """

# =======================================================
# 4. AGENTE ESPECIALISTA (Microempresa)
# =======================================================
micro_agent = Agent(
    model=bedrock_model,
    deps_type=LegalDeps
)

# ADICIONADO: Prompt dinâmico que recebe o histórico
@micro_agent.system_prompt
def prompt_micro(ctx: RunContext[LegalDeps]) -> str:
    return f"""
    Você é um advogado especialista em Microempresas, EPP e Simples Nacional (LC 123/2006).
    Use a ferramenta de busca para fundamentar suas respostas.
    
    IMPORTANTE:
    Use o histórico abaixo para manter o contexto da conversa (ex: se o usuário disser "e quais os impostos?", saiba do que ele está falando).
    
    Histórico da Conversa:
    {ctx.deps.historico_conversa}
    """

@micro_agent.tool
def tool_buscar_leis_micro(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    """Busca leis usando a engine do LlamaIndex."""
    response = ctx.deps.query_engine.query(termo_busca)
    return str(response)

# =======================================================
# 5. AGENTE ESPECIALISTA (Geral/Grande)
# =======================================================
general_agent = Agent(
    model=bedrock_model,
    deps_type=LegalDeps
)

@general_agent.system_prompt
def prompt_geral(ctx: RunContext[LegalDeps]) -> str:
    return f"""
    Você é um consultor jurídico sênior focado em Médias e Grandes Empresas (S.A., Lucro Real).
    Foque em Compliance e Governança.
    
    IMPORTANTE:
    Use o histórico abaixo para manter o contexto da conversa.
    
    Histórico da Conversa:
    {ctx.deps.historico_conversa}
    """

@general_agent.tool
def tool_buscar_leis_geral(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    """Busca leis usando a engine do LlamaIndex."""
    response = ctx.deps.query_engine.query(termo_busca)
    return str(response)