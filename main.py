import logging
from dataclasses import dataclass, field
from typing import List

import Prompts
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.base import BaseCheckpointSaver
import Agents
from pydantic import BaseModel, Field

from Prompts import juiz_tmpl
from LLM import sonnet_bedrock_model

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
    judge_feedback: str = ""
    revision_count: int = 0
    is_approved: bool = False

# =======================================================
# 2. HELPER (Auxiliar para atualizar memória)
# =======================================================
def _atualizar_historico(state: WorkflowState, resposta_ai: str) -> List[str]:
    """
    Pega o histórico antigo e adiciona a interação atual.
    O LangGraph salvará essa nova lista no Postgres automaticamente.
    """
    nova_interacao = [
        f"User: {state.user_question}",
        f"AI: {resposta_ai}"
    ]
    return state.chat_history + nova_interacao

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
    
    pergunta = state.user_question
    if state.judge_feedback and state.revision_count > 0:
        logging.info(f"🔄 Refazendo resposta Tributária (Tentativa {state.revision_count})...")
        pergunta += f"\n\n[INSTRUÇÃO DO AUDITOR PARA CORREÇÃO]:\n{state.judge_feedback}"

    result = await Agents.tributario_agent.run(pergunta, deps=deps)
    
    return {"final_response": str(result.output)}

async def node_trabalhista(state: WorkflowState):
    logging.info("--- AGENTE: Trabalhista (CLT) ---")
    deps = _preparar_dependencias(state)
    
    pergunta = state.user_question
    if state.judge_feedback and state.revision_count > 0:
        logging.info(f"🔄 Refazendo resposta Trabalhista (Tentativa {state.revision_count})...")
        pergunta += f"\n\n[INSTRUÇÃO DO AUDITOR PARA CORREÇÃO]:\n{state.judge_feedback}"

    result = await Agents.trabalhista_agent.run(pergunta, deps=deps)
    
    return {"final_response": str(result.output)}

async def node_societario(state: WorkflowState):
    logging.info("--- AGENTE: Societario (Burocracia / Lei 14.195) ---")
    deps = _preparar_dependencias(state)
    
    pergunta = state.user_question
    if state.judge_feedback and state.revision_count > 0:
        logging.info(f"🔄 Refazendo resposta Societária (Tentativa {state.revision_count})...")
        pergunta += f"\n\n[INSTRUÇÃO DO AUDITOR PARA CORREÇÃO]:\n{state.judge_feedback}"

    result = await Agents.societario_agent.run(pergunta, deps=deps)
    
    return {"final_response": str(result.output)}

async def node_conversational(state: WorkflowState):
    logging.info("--- AGENTE: Conversational (Papo Social) ---")
    deps = _preparar_dependencias(state)
    result = await Agents.conversational_agent.run(state.user_question, deps=deps)
    
    resp = str(result.output)
    
    return {
        "final_response": resp,
        "chat_history": _atualizar_historico(state, resp)
    }

async def node_out_of_scope(state: WorkflowState):
    resp = (
        "Desculpe, não posso ajudar com esse assunto.\n\n"
        "Minha base de conhecimento é restrita e especializada apenas em:\n"
        "1. **Tributário:** Simples Nacional e Pronampe (LC 123/2006 e Lei 13.999);\n"
        "2. **Trabalhista:** Regras da CLT e contratações;\n"
        "3. **Societário:** Abertura de empresas e Lei do Ambiente de Negócios (Lei 14.195).\n\n"
        "Para outros temas jurídicos (como Criminal, Família, ou Falências) ou assuntos gerais, não tenho informações disponíveis."
    )

    return {
        "final_response": resp,
        "chat_history": _atualizar_historico(state, resp)
    }
    

async def node_juiz(state: WorkflowState):
    logging.info("--- AGENTE: Juiz (Avaliando Resposta) ---")
    deps = _preparar_dependencias(state)
    
    texto_pronto_para_o_juiz = Prompts.juiz_tmpl.format(
        user_question=state.user_question,
        final_response=state.final_response
    )
    
    # O Agents.judge_agent deve estar configurado com result_type=AvaliacaoJuiz
    result = await Agents.judge_agent.run(texto_pronto_para_o_juiz, deps=deps)
    veredito = result.output
    
    # Critério rígido de aprovação
    aprovado = veredito.nota >= 4 and not veredito.tem_alucinacao
    revisoes = state.revision_count + 1
    
    logging.info(f"⚖️ NOTA DO JUIZ: {veredito.nota}/5 | Aprovado: {aprovado} | Motivo: {veredito.justificativa}")
    
    # Só atualiza o histórico se foi aprovado ou se já esgotou as tentativas
    novo_historico = state.chat_history
    if aprovado or revisoes >= 2:
        novo_historico = _atualizar_historico(state, state.final_response)
        if revisoes >= 2 and not aprovado:
            logging.warning("⚠️ Limite de 2 revisões atingido. Enviando resposta como está.")

    return {
        "judge_feedback": veredito.correcao_necessaria if not aprovado else "",
        "is_approved": aprovado,
        "revision_count": revisoes,
        "chat_history": novo_historico
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
    
def rotear_pos_juiz(state: WorkflowState):
    """Verifica se encerra ou devolve para o agente que errou."""
    if state.is_approved or state.revision_count >= 2:
        return "end"
    
    # Se não foi aprovado, devolve para a caixinha certa
    if state.classification_profile == TRIBUTARIO: 
        return TRIBUTARIO
    elif state.classification_profile == TRABALHISTA: 
        return TRABALHISTA
    elif state.classification_profile == SOCIETARIO: 
        return SOCIETARIO
    
    return "end" # Fallback de segurança

def create_workflow(query_engine, checkpointer: BaseCheckpointSaver = None):
    global _engine_instance
    _engine_instance = query_engine

    NODE_ROUTER = "node_router"
    NODE_TRIBUTARIO = f"node_{TRIBUTARIO}"
    NODE_TRABALHISTA = f"node_{TRABALHISTA}"
    NODE_SOCIETARIO = f"node_{SOCIETARIO}"
    NODE_CONVERSATIONAL = f"node_{CONVERSATIONAL}"
    NODE_OUT_OF_SCOPE = f"node_{OUT_OF_SCOPE}"
    NODE_JUIZ = "node_juiz"
    
    workflow = StateGraph(WorkflowState)
    workflow.add_node(NODE_ROUTER, node_router)
    workflow.add_node(NODE_TRIBUTARIO, node_tributario)
    workflow.add_node(NODE_TRABALHISTA, node_trabalhista)
    workflow.add_node(NODE_SOCIETARIO, node_societario)
    workflow.add_node(NODE_CONVERSATIONAL, node_conversational)
    workflow.add_node(NODE_OUT_OF_SCOPE, node_out_of_scope)
    workflow.add_node(NODE_JUIZ, node_juiz)
    
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
    
    workflow.add_edge(NODE_TRIBUTARIO, NODE_JUIZ)
    workflow.add_edge(NODE_TRABALHISTA, NODE_JUIZ)
    workflow.add_edge(NODE_SOCIETARIO, NODE_JUIZ)
    workflow.add_edge(NODE_CONVERSATIONAL, END)
    workflow.add_edge(NODE_OUT_OF_SCOPE, END)

    mapa_juiz = {
        TRIBUTARIO: NODE_TRIBUTARIO,
        TRABALHISTA: NODE_TRABALHISTA,
        SOCIETARIO: NODE_SOCIETARIO,
        "end": END
    }

    workflow.add_conditional_edges(NODE_JUIZ, rotear_pos_juiz, mapa_juiz)
    
    return workflow.compile(checkpointer=checkpointer)