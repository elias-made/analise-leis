import logging
from dataclasses import dataclass, field
from typing import List

from langgraph.graph import StateGraph, END, START

import Agents

TRIBUTARIO = "tributario"
TRABALHISTA = "trabalhista"
SOCIETARIO = "societario"
CONVERSATIONAL = "conversational"
OUT_OF_SCOPE = "out_of_scope"

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

def _preparar_dependencias(state: WorkflowState) -> Agents.LegalDeps:
    historico_str = "\n".join(state.chat_history)
    return Agents.LegalDeps(
        query_engine=_engine_instance,
        historico_conversa=historico_str
    )

async def node_router(state: WorkflowState):
    logging.info("--- ROUTER: Classificando perfil ---")
    deps = _preparar_dependencias(state)
    result = await Agents.router_agent.run(state.user_question, deps=deps)
    
    raw_response = str(result.output).strip().lower()
    profile = raw_response.split()[0].replace(".", "").replace(",", "")
    
    profiles = [TRIBUTARIO, TRABALHISTA, SOCIETARIO, CONVERSATIONAL, OUT_OF_SCOPE]

    if profile not in profiles:
       logging.warning(f"Router retornou classificação inválida: '{profile}'. Redirecionando para OUT_OF_SCOPE.")
       profile = OUT_OF_SCOPE
    
    logging.info(f"--- ROUTER: Perfil definido como '{profile}' ---")
    return {"classification_profile": profile}

async def node_tributario(state: WorkflowState):
    logging.info("--- AGENTE: Simples Nacional, ME/EPP e Pronampe ---")
    deps = _preparar_dependencias(state)
    result = await Agents.tributario_agent.run(state.user_question, deps=deps)
    return {"final_response": str(result.output)}

async def node_trabalhista(state: WorkflowState):
    logging.info("--- AGENTE: Trabalhista (CLT) ---")
    deps = _preparar_dependencias(state)
    result = await Agents.trabalhista_agent.run(state.user_question, deps=deps)
    return {"final_response": str(result.output)}

async def node_societario(state: WorkflowState):
    logging.info("--- AGENTE: Societario (Burocracia / Lei 14.195) ---")
    deps = _preparar_dependencias(state)
    result = await Agents.societario_agent.run(state.user_question, deps=deps)
    return {"final_response": str(result.output)}

async def node_conversational(state: WorkflowState):
    logging.info("--- AGENTE: Conversational (Papo Social) ---")
    deps = _preparar_dependencias(state)
    result = await Agents.conversational_agent.run(state.user_question, deps=deps)
    return {"final_response": str(result.output)}

async def node_out_of_scope(state: WorkflowState):
    return {
        "final_response": 
        "Desculpe, não posso ajudar com esse assunto.\n\n"
        "Minha base de conhecimento é restrita e especializada apenas em:\n"
        "1. **Tributário:** Simples Nacional e Pronampe (LC 123/2006 e Lei 13.999);\n"
        "2. **Trabalhista:** Regras da CLT e contratações;\n"
        "3. **Societário:** Abertura de empresas e Lei do Ambiente de Negócios (Lei 14.195).\n\n"
        "Para outros temas jurídicos (como Criminal, Família, ou Falências) ou assuntos gerais, não tenho informações disponíveis."
    }

# =======================================================
# 3. LÓGICA E CRIAÇÃO
# =======================================================
def check_profile_logic(state: WorkflowState):
    if state.classification_profile == TRIBUTARIO:
        return TRIBUTARIO
    elif state.classification_profile == TRABALHISTA:
        return TRABALHISTA
    elif state.classification_profile == SOCIETARIO:
        return SOCIETARIO
    elif state.classification_profile == CONVERSATIONAL:
        return CONVERSATIONAL
    else:
        return OUT_OF_SCOPE

def create_workflow(query_engine):
    global _engine_instance
    _engine_instance = query_engine

    NODE_ROUTER = "node_router"
    NODE_TRIBUTARIO = f"node_{TRIBUTARIO}"
    NODE_TRABALHISTA = f"node_{TRABALHISTA}"
    NODE_SOCIETARIO = f"node_{SOCIETARIO}"
    NODE_CONVERSATIONAL = f"node_{CONVERSATIONAL}"
    NODE_OUT_OF_SCOPE = f"node_{OUT_OF_SCOPE}"
    
    workflow = StateGraph(WorkflowState)
    workflow.add_node(NODE_ROUTER, node_router)
    workflow.add_node(NODE_TRIBUTARIO, node_tributario)
    workflow.add_node(NODE_TRABALHISTA, node_trabalhista)
    workflow.add_node(NODE_SOCIETARIO, node_societario)
    workflow.add_node(NODE_CONVERSATIONAL, node_conversational)
    workflow.add_node(NODE_OUT_OF_SCOPE, node_out_of_scope)
    
    workflow.add_edge(START, NODE_ROUTER)

    mapa_decisao = {
        TRIBUTARIO: NODE_TRIBUTARIO,
        TRABALHISTA: NODE_TRABALHISTA,
        SOCIETARIO: NODE_SOCIETARIO,
        CONVERSATIONAL: NODE_CONVERSATIONAL,
        OUT_OF_SCOPE: NODE_OUT_OF_SCOPE
    }
    
    workflow.add_conditional_edges(
        NODE_ROUTER,
        check_profile_logic,
        mapa_decisao
    )
    
    workflow.add_edge(NODE_TRIBUTARIO, END)
    workflow.add_edge(NODE_TRABALHISTA, END)
    workflow.add_edge(NODE_SOCIETARIO, END)
    workflow.add_edge(NODE_CONVERSATIONAL, END)
    workflow.add_edge(NODE_OUT_OF_SCOPE, END)
    
    return workflow.compile()