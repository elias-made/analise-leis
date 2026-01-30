import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List

from langgraph.graph import StateGraph, END, START

from Agents import (
    router_agent, 
    micro_agent, 
    general_agent, 
    LegalDeps
)

# =======================================================
# 1. ESTADO DO WORKFLOW
# =======================================================
@dataclass
class WorkflowState:
    user_question: str
    chat_history: List[str] = field(default_factory=list)
    classification_profile: str = None
    final_response: str = None

# =======================================================
# 2. IMPLEMENTAÇÃO DOS NÓS
# =======================================================
_engine_instance = None

def _preparar_dependencias(state: WorkflowState) -> LegalDeps:
    historico_str = "\n".join(state.chat_history)
    return LegalDeps(
        query_engine=_engine_instance,
        historico_conversa=historico_str
    )

async def node_router(state: WorkflowState):
    logging.info("--- ROUTER: Classificando perfil ---")
    deps = _preparar_dependencias(state)
    result = await router_agent.run(state.user_question, deps=deps)
    
    raw_response = str(result.output).strip().lower()
    profile = raw_response.split()[0].replace(".", "").replace(",", "")
    
    if profile not in ['micro', 'media', 'grande', 'geral']:
        profile = 'geral'
    
    logging.info(f"--- ROUTER: Perfil definido como '{profile}' ---")
    return {"classification_profile": profile}

async def node_micro(state: WorkflowState):
    logging.info("--- AGENTE: Microempresa ---")
    deps = _preparar_dependencias(state)
    result = await micro_agent.run(state.user_question, deps=deps)
    return {"final_response": str(result.output)}

async def node_geral(state: WorkflowState):
    logging.info("--- AGENTE: Geral/Grande ---")
    deps = _preparar_dependencias(state)
    result = await general_agent.run(state.user_question, deps=deps)
    return {"final_response": str(result.output)}

# =======================================================
# 3. LÓGICA E CRIAÇÃO
# =======================================================
def check_profile_logic(state: WorkflowState):
    if state.classification_profile == 'micro':
        return "node_micro"
    else:
        return "node_geral"

def create_workflow(query_engine):
    global _engine_instance
    _engine_instance = query_engine
    
    workflow = StateGraph(WorkflowState)
    workflow.add_node("node_router", node_router)
    workflow.add_node("node_micro", node_micro)
    workflow.add_node("node_geral", node_geral)
    
    workflow.add_edge(START, "node_router")
    
    workflow.add_conditional_edges(
        "node_router",
        check_profile_logic,
        {"node_micro": "node_micro", "node_geral": "node_geral"}
    )
    
    workflow.add_edge("node_micro", END)
    workflow.add_edge("node_geral", END)
    
    return workflow.compile()