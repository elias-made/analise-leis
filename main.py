import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any

import Prompts
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.base import BaseCheckpointSaver
import Agents
from langchain_core.runnables import RunnableConfig

import asyncio
import fitz

from langfuse import Langfuse
from utils import corrigir_formatacao_markdown, ler_pdf_bytes, preparar_historico_estruturado
langfuse_client = Langfuse()

SIMPLES = "simples"
TRABALHISTA = "trabalhista"
SOCIETARIO = "societario"
CORPORATIVO = "corporativo"
CONVERSATIONAL = "conversational"
OUT_OF_SCOPE = "out_of_scope"
LIMPEZA = "limpeza"

# =======================================================
# 1. ESTADO DO WORKFLOW
# =======================================================
@dataclass
class WorkflowState:
    user_question: str
    file_bytes: Optional[bytes] = None
    document_content: str = ""
    chat_history: List[str] = field(default_factory=list)
    classification_profile: str = None
    final_response: str = None

# =======================================================
# 2. HELPER (Auxiliar para atualizar memória)
# =======================================================
def _atualizar_historico(state: WorkflowState, resposta_ai: str) -> List[str]:
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
        historico_conversa=historico_limpo,
        documento_texto=state.document_content or "Nenhum documento anexado."
    )

async def node_leitor(state: WorkflowState):
    logging.info("--- NODE: Leitor de Documentos ---")
    
    # Se não tem arquivo novo, mantém o que já estava no estado
    if not state.file_bytes:
        if state.document_content:
            logging.info("Mantendo contexto anterior.")
        return {} 

    logging.info("Processando novo arquivo PDF...")

    # AQUI ESTÁ A MÁGICA: Chamamos a função auxiliar
    texto_processado = ler_pdf_bytes(state.file_bytes)
    
    logging.info(f"Leitura concluída. Tamanho: {len(texto_processado)} chars.")

    return {
        "document_content": texto_processado,
        # Importante: Zeramos o binário para não pesar no banco de dados
        "file_bytes": None 
    }

async def node_router(state: WorkflowState):
    logging.info("--- ROUTER: Classificando perfil ---")
    deps = _preparar_dependencias(state)
    
    result = await Agents.router_agent.run(state.user_question, deps=deps)
    raw_profile = str(result.output)

    profile = raw_profile.replace("`", "").replace("'", "").replace('"', "").replace('*', "").strip().lower()
    
    logging.info(f"--- ROUTER: Perfil definido como '{profile}' ---")
    perfis_validos = [SIMPLES, TRABALHISTA, SOCIETARIO, CORPORATIVO, CONVERSATIONAL, OUT_OF_SCOPE]
    if profile not in perfis_validos:
        logging.warning(f"Router retornou perfil desconhecido: '{profile}'. Redirecionando para OUT_OF_SCOPE.")
        profile = OUT_OF_SCOPE

    return {"classification_profile": profile}

async def node_simples(state: WorkflowState):
    logging.info("--- AGENTE: Simples Nacional, ME/EPP e Pronampe ---")
    deps = _preparar_dependencias(state)
    
    pergunta = state.user_question

    result = await Agents.simples_agent.run(pergunta, deps=deps)
    
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

async def node_corporativo(state: WorkflowState):
    logging.info("--- AGENTE: Corporativo (S/A e Lucro Real) ---")
    deps = _preparar_dependencias(state)
    # Chama o agente novo definido no Agents.py
    result = await Agents.corporativo_agent.run(state.user_question, deps=deps)
    return {"final_response": str(result.output)}

async def node_limpeza(state: WorkflowState):
    logging.info("--- NODE: Limpeza e Padronização ---")
    
    resposta_suja = state.final_response
    if not resposta_suja:
        return {}

    resposta_limpa = corrigir_formatacao_markdown(resposta_suja)
    
    return {"final_response": resposta_limpa}

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
        "Minha base de conhecimento é restrita a:\n"
        "1. **Tributário:** Simples Nacional, Lucro Presumido e Real;\n"
        "2. **Corporativo:** Sociedades Anônimas (S/A) e Governança;\n"
        "3. **Trabalhista:** Regras da CLT;\n"
        "4. **Societário:** Abertura e gestão de Limitadas (LTDA).\n\n"
        "Para assuntos como Penal, Família ou Previdenciário (Pessoa Física), não tenho informações."
    )

    return {
        "final_response": resp,
        "chat_history": _atualizar_historico(state, resp)
    }
    
# =======================================================
# TRABALHADOR EM BACKGROUND (ASYNC)
# =======================================================
async def _auditoria_background(user_question, final_response, chat_history, profile, deps_query_engine, historico_str, session_id):
    """
    Função puramente assíncrona. Não precisa de Thread nem asyncio.run
    """
    try:
        # Pequeno sleep para garantir que a resposta já foi enviada ao usuário
        # e liberar o event loop para o Frontend gerar o título
        await asyncio.sleep(0.5) 

        deps = Agents.LegalDeps(query_engine=deps_query_engine, historico_conversa=historico_str)
        
        texto_pronto = Prompts.juiz_tmpl.format(
            historico=historico_str,
            user_question=user_question,
            final_response=final_response
        )
        
        # Aqui ele vai aguardar o LLM, mas como é await, ele libera o processador
        # para outras coisas (como gerar o título) enquanto espera a API responder
        result = await Agents.judge_agent.run(texto_pronto, deps=deps)
        
        m = result.output.metricas
        sugestao = result.output.correcao_necessaria
        
        historico_visual = preparar_historico_estruturado(chat_history)

        trace = langfuse_client.trace(
            name="Auditoria_Juiz",
            session_id=session_id,
            input={
                "pergunta_usuario": user_question,
                "contexto_anterior": historico_visual,
                "resposta_avaliada": final_response,
            },
            output=sugestao,
            tags=[profile or "geral"]
        )
        
        trace.score(name="Fundamentacao", value=m.fundamentacao)
        trace.score(name="Utilidade", value=m.utilidade)
        trace.score(name="Tom_de_Voz", value=m.tom_de_voz)
        trace.score(name="Protocolo_Visual", value=m.protocolo_visual)
        
        langfuse_client.flush()
        logging.info("--- JUIZ: Auditoria concluída em background ---")
        
    except Exception as e:
        logging.error(f"Erro na Task do Juiz: {e}")


# =======================================================
# NÓ DO GRAFO
# =======================================================
async def node_juiz(state: WorkflowState, config: RunnableConfig = None):
    logging.info("--- AGENTE: Juiz (Disparando Thread Paralela) ---")
    
    session_id = "sessao_padrao"
    if config and "configurable" in config:
        session_id = config["configurable"].get("thread_id", "sessao_padrao")
    
    novo_historico = _atualizar_historico(state, state.final_response)
    historico_str = "\n".join(state.chat_history) if state.chat_history else "Nenhuma conversa anterior."

    asyncio.create_task(
        _auditoria_background(
            state.user_question,
            state.final_response,
            state.chat_history,
            state.classification_profile,
            _engine_instance,
            historico_str,
            session_id
        )
    )

    return {
        "chat_history": novo_historico
    }


# =======================================================
# 3. LÓGICA E CRIAÇÃO
# =======================================================
def check_profile_logic(state: WorkflowState):
    if state.classification_profile == SIMPLES:
        return SIMPLES
    elif state.classification_profile == TRABALHISTA:
        return TRABALHISTA
    elif state.classification_profile == SOCIETARIO:
        return SOCIETARIO
    elif state.classification_profile == CORPORATIVO:
        return CORPORATIVO
    elif state.classification_profile == CONVERSATIONAL:
        return CONVERSATIONAL
    else:
        return OUT_OF_SCOPE

def create_workflow(query_engine, checkpointer: BaseCheckpointSaver = None):
    global _engine_instance
    _engine_instance = query_engine

    NODE_LEITOR = "node_leitor"
    NODE_ROUTER = "node_router"
    NODE_SIMPLES = f"node_{SIMPLES}"
    NODE_TRABALHISTA = f"node_{TRABALHISTA}"
    NODE_SOCIETARIO = f"node_{SOCIETARIO}"
    NODE_CORPORATIVO = f"node_{CORPORATIVO}"
    NODE_LIMPEZA = "node_limpeza"
    NODE_CONVERSATIONAL = f"node_{CONVERSATIONAL}"
    NODE_OUT_OF_SCOPE = f"node_{OUT_OF_SCOPE}"
    NODE_JUIZ = "node_juiz"
    
    workflow = StateGraph(WorkflowState)
    workflow.add_node(NODE_LEITOR, node_leitor)
    workflow.add_node(NODE_ROUTER, node_router)

    workflow.add_node(NODE_SIMPLES, node_simples)
    workflow.add_node(NODE_TRABALHISTA, node_trabalhista)
    workflow.add_node(NODE_SOCIETARIO, node_societario)
    workflow.add_node(NODE_CORPORATIVO, node_corporativo)

    workflow.add_node(NODE_LIMPEZA, node_limpeza)

    workflow.add_node(NODE_CONVERSATIONAL, node_conversational)
    workflow.add_node(NODE_OUT_OF_SCOPE, node_out_of_scope)
    workflow.add_node(NODE_JUIZ, node_juiz)
    
    workflow.add_edge(START, NODE_LEITOR)
    workflow.add_edge(NODE_LEITOR, NODE_ROUTER)

    mapa_decisao = {
        SIMPLES: NODE_SIMPLES,
        TRABALHISTA: NODE_TRABALHISTA,
        SOCIETARIO: NODE_SOCIETARIO,
        CORPORATIVO: NODE_CORPORATIVO,
        CONVERSATIONAL: NODE_CONVERSATIONAL,
        OUT_OF_SCOPE: NODE_OUT_OF_SCOPE
    }
    
    workflow.add_conditional_edges(
        NODE_ROUTER,
        check_profile_logic,
        mapa_decisao
    )
    
    workflow.add_edge(NODE_SIMPLES, NODE_LIMPEZA)
    workflow.add_edge(NODE_TRABALHISTA, NODE_LIMPEZA)
    workflow.add_edge(NODE_SOCIETARIO, NODE_LIMPEZA)
    workflow.add_edge(NODE_CORPORATIVO, NODE_LIMPEZA)

    workflow.add_edge(NODE_LIMPEZA, NODE_JUIZ)

    workflow.add_edge(NODE_JUIZ, END)
    workflow.add_edge(NODE_CONVERSATIONAL, END)
    workflow.add_edge(NODE_OUT_OF_SCOPE, END)
    
    return workflow.compile(checkpointer=checkpointer)