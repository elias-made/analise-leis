import logging
from dataclasses import dataclass, field
from typing import List

import Prompts
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.base import BaseCheckpointSaver
import Agents
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

import threading
import asyncio

from langfuse import Langfuse
from utils import preparar_historico_estruturado
langfuse_client = Langfuse()

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
    # Em vez de criar uma string gigante com .join("\n"), 
    # criamos a lista de objetos estruturada.
    historico_limpo = preparar_historico_estruturado(state.chat_history)
    
    return Agents.LegalDeps(
        query_engine=_engine_instance,
        historico_conversa=historico_limpo # Enviamos a lista de dicts
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

    result = await Agents.tributario_agent.run(pergunta, deps=deps)
    
    return {"final_response": str(result.output)}

async def node_trabalhista(state: WorkflowState):
    logging.info("--- AGENTE: Trabalhista (CLT) ---")
    deps = _preparar_dependencias(state)
    
    pergunta = state.user_question

    result = await Agents.trabalhista_agent.run(pergunta, deps=deps)
    
    return {"final_response": str(result.output)}

async def node_societario(state: WorkflowState):
    logging.info("--- AGENTE: Societario (Burocracia / Lei 14.195) ---")
    deps = _preparar_dependencias(state)
    
    pergunta = state.user_question

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
    
# =======================================================
# TRABALHADOR PARALELO (Imune ao Streamlit)
# =======================================================
def _auditoria_thread(user_question, final_response, chat_history, profile, deps_query_engine, historico_str, session_id):
    async def _run_async_judge():
        try:
            deps = Agents.LegalDeps(query_engine=deps_query_engine, historico_conversa=historico_str)
            
            texto_pronto = Prompts.juiz_tmpl.format(
                historico=historico_str,
                user_question=user_question,
                final_response=final_response
            )
            
            result = await Agents.judge_agent.run(texto_pronto, deps=deps)
            m = result.output.metricas
            
            # FORMATANDO PARA O LANGFUSE (O visual 'legal' que você queria)
            historico_visual = preparar_historico_estruturado(chat_history)

            trace = langfuse_client.trace(
                name="Auditoria_Juiz",
                session_id=session_id,
                input={
                    "pergunta_usuario": user_question,
                    "contexto_anterior": historico_visual, 
                },
                output=final_response,
                tags=[profile or "geral"]
            )
            
            # ... resto do código de scores (Fundamentação, Utilidade, etc) ...
            trace.score(name="Fundamentacao", value=m.fundamentacao)
            trace.score(name="Utilidade", value=m.utilidade)
            trace.score(name="Protocolo_Visual", value=m.protocolo_visual)
            trace.score(name="Tom_de_Voz", value=m.tom_de_voz)
            
            langfuse_client.flush()
        except Exception as e:
            logging.error(f"Erro na Thread do Juiz: {e}")

    asyncio.run(_run_async_judge())


# =======================================================
# NÓ DO GRAFO (Ultra rápido)
# =======================================================
async def node_juiz(state: WorkflowState, config: RunnableConfig = None):
    logging.info("--- AGENTE: Juiz (Disparando Thread Paralela) ---")
    
    # LÓGICA DE SEGURANÇA: Garante que temos um session_id mesmo sem config
    session_id = "sessao_padrao"
    if config and "configurable" in config:
        session_id = config["configurable"].get("thread_id", "sessao_padrao")
    
    # 1. Atualiza o histórico imediatamente
    novo_historico = _atualizar_historico(state, state.final_response)
    historico_str = "\n".join(state.chat_history) if state.chat_history else "Nenhuma conversa anterior."

    # 2. CHUTE INVISÍVEL REAL: Cria uma Thread nativa do Sistema Operacional
    threading.Thread(
        target=_auditoria_thread,
        args=(
            state.user_question,
            state.final_response,
            state.chat_history,
            state.classification_profile,
            _engine_instance,
            historico_str,
            session_id # Passamos o ID seguro
        ),
        daemon=True 
    ).start()

    # 3. Retorna instantaneamente
    return {
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

    workflow.add_edge(NODE_JUIZ, END)
    
    return workflow.compile(checkpointer=checkpointer)