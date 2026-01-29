import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END, START

# --- PydanticAI Imports ---
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockModel

# --- LlamaIndex Imports ---
# Importamos apenas a classe base para tipagem
from llama_index.core.base.base_query_engine import BaseQueryEngine

# =======================================================
# 1. ESTADO DO WORKFLOW (State com Dataclass)
# =======================================================
@dataclass
class WorkflowState:
    """
    Define a estrutura de dados que trafega pelo grafo.
    Similar ao 'workflowState' do exemplo financeiro.
    """
    # Entrada: A pergunta atual do usuário
    user_question: str
    
    # Memória: Lista de mensagens anteriores (ex: ["User: Oi", "AI: Olá"])
    # default_factory=list garante que comece vazio se não for passado
    chat_history: List[str] = field(default_factory=list)
    
    # Variáveis internas de controle (Preenchidas pelos nós)
    classification_profile: str = None  # 'micro', 'media', 'grande' ou 'geral'
    final_response: str = None          # A resposta final gerada

# =======================================================
# 2. CONFIGURAÇÃO PYDANTIC AI (Dependências e Modelos)
# =======================================================

@dataclass
class LegalDeps:
    """Dependências injetadas dentro dos Agentes PydanticAI"""
    query_engine: BaseQueryEngine
    contexto_historico: str  # Histórico formatado como string única

# Configuração do Modelo AWS Bedrock (Claude 3.5 Sonnet)
# O PydanticAI usa boto3 internamente para autenticar
bedrock_model = BedrockModel(
    model='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    aws_region='us-east-1'
)

# =======================================================
# 3. DEFINIÇÃO DOS AGENTES (A "Inteligência")
# =======================================================

# --- AGENTE 1: ROUTER (Classificador) ---
OutputPerfil = Literal['micro', 'media', 'grande', 'geral']

router_agent = Agent(
    model=bedrock_model,
    result_type=OutputPerfil, # Garante retorno estruturado e validado
    deps_type=LegalDeps,
    system_prompt=(
        "Você é um classificador de intenção jurídica. "
        "Analise o histórico e a última pergunta para determinar o porte da empresa. "
        "Se o usuário mencionar 'ME', 'EPP', 'Simples', retorne 'micro'. "
        "Se mencionar 'S.A.', 'Compliance', 'Multinacional', retorne 'grande'. "
        "Se não estiver claro, retorne 'geral'."
    )
)

# --- AGENTE 2: ESPECIALISTA MICROEMPRESA ---
micro_agent = Agent(
    model=bedrock_model,
    deps_type=LegalDeps,
    system_prompt=(
        "Você é um advogado especialista em Microempresas e EPP (Lei Complementar 123/Simples Nacional). "
        "Use o histórico da conversa para manter o contexto. "
        "Se precisar de embasamento legal, use a ferramenta de busca."
    )
)

@micro_agent.tool
def tool_buscar_leis_micro(ctx: RunContext[LegalDeps], query: str) -> str:
    """Busca leis e artigos no banco de dados vetorial."""
    # Acessa o engine do LlamaIndex injetado
    return str(ctx.deps.query_engine.query(query))

# --- AGENTE 3: ESPECIALISTA GERAL / GRANDE ---
general_agent = Agent(
    model=bedrock_model,
    deps_type=LegalDeps,
    system_prompt=(
        "Você é um consultor jurídico sênior focado em Médias e Grandes Empresas (S.A., Lucro Real). "
        "Foque em Compliance, Governança e legislação complexa. "
        "Use o histórico da conversa para manter o contexto. "
        "Se precisar de embasamento legal, use a ferramenta de busca."
    )
)

@general_agent.tool
def tool_buscar_leis_geral(ctx: RunContext[LegalDeps], query: str) -> str:
    """Busca leis e artigos no banco de dados vetorial."""
    return str(ctx.deps.query_engine.query(query))

# =======================================================
# 4. IMPLEMENTAÇÃO DOS NÓS (LangGraph Nodes)
# =======================================================

# Variável global para injetar o engine (Padrão Factory)
_engine_instance = None

def _preparar_dependencias(state: WorkflowState) -> LegalDeps:
    """Helper para formatar o histórico e criar as deps."""
    historico_str = "\n".join(state.chat_history)
    return LegalDeps(
        query_engine=_engine_instance,
        contexto_historico=historico_str
    )

async def node_router(state: WorkflowState):
    """
    Passo 1: Recebe o estado, formata o histórico e decide o perfil.
    """
    logging.info("--- ROUTER: Classificando perfil ---")
    
    deps = _preparar_dependencias(state)
    
    # O input do agente é a pergunta atual do usuário
    # O contexto histórico vai via 'deps'
    result = await router_agent.run(state.user_question, deps=deps)
    
    logging.info(f"--- ROUTER: Perfil definido como '{result.data}' ---")
    
    # Atualiza apenas o campo de perfil no estado
    return {"classification_profile": result.data}

async def node_micro(state: WorkflowState):
    """
    Passo 2A: Executa o especialista em Microempresas.
    """
    logging.info("--- AGENTE: Microempresa ---")
    
    deps = _preparar_dependencias(state)
    result = await micro_agent.run(state.user_question, deps=deps)
    
    # Retorna a resposta final
    return {"final_response": result.data}

async def node_geral(state: WorkflowState):
    """
    Passo 2B: Executa o especialista Geral/Grande.
    """
    logging.info("--- AGENTE: Geral/Grande ---")
    
    deps = _preparar_dependencias(state)
    result = await general_agent.run(state.user_question, deps=deps)
    
    return {"final_response": result.data}

# =======================================================
# 5. LÓGICA DE CONDICIONAL E CRIAÇÃO DO WORKFLOW
# =======================================================

def check_profile_logic(state: WorkflowState):
    """Lógica de decisão: Para qual nó vamos agora?"""
    perfil = state.classification_profile
    
    if perfil == 'micro':
        return "node_micro"
    else:
        # 'media', 'grande' e 'geral' caem aqui (pode expandir se quiser)
        return "node_geral"

def create_workflow(query_engine: BaseQueryEngine):
    """
    Função principal chamada pela API para montar o 'cérebro'.
    """
    global _engine_instance
    _engine_instance = query_engine # Guarda referência do LlamaIndex
    
    # Inicializa o Grafo tipado com nosso Dataclass
    workflow = StateGraph(WorkflowState)
    
    # A. Adiciona os Nós
    workflow.add_node("node_router", node_router)
    workflow.add_node("node_micro", node_micro)
    workflow.add_node("node_geral", node_geral)
    
    # B. Define o Início
    workflow.add_edge(START, "node_router")
    
    # C. Define as Arestas Condicionais
    workflow.add_conditional_edges(
        "node_router",        # Origem
        check_profile_logic,  # Função de decisão
        {
            "node_micro": "node_micro",
            "node_geral": "node_geral"
        }
    )
    
    # D. Define o Fim (Os agentes especialistas encerram o fluxo)
    workflow.add_edge("node_micro", END)
    workflow.add_edge("node_geral", END)
    
    # E. Compila
    app = workflow.compile()
    
    return app