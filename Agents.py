import os
from typing import List
import redis
import numpy as np
import hashlib
from dataclasses import dataclass

# Imports do Redis Stack
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from pydantic_ai import Agent, RunContext
from llama_index.core.base.base_query_engine import BaseQueryEngine
from pydantic import BaseModel, Field

from ddgs import DDGS
from datetime import datetime

import Prompts
from LLM import (
    sonnet_bedrock_model,
    embed_model 
)

# =======================================================
# 1. SISTEMA DE CACHE SEMÂNTICO (REDIS STACK) 🧠
# =======================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
INDEX_NAME = "idx:rag_cache_v1"
REAL_VECTOR_DIM = 1024 
USE_REDIS = False
_redis_client = None

def gerar_hash_estavel(texto: str) -> str:
    return hashlib.md5(texto.encode('utf-8')).hexdigest()

try:
    redis_url = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/0")
    _redis_client = redis.Redis.from_url(redis_url, decode_responses=False) 
    _redis_client.ping()
    print(f"✅ REDIS STACK: Conectado (Modo Binário).")
    USE_REDIS = True

    print("📏 Medindo dimensão do embedding...")
    test_vec = embed_model.get_query_embedding("teste de dimensao")
    REAL_VECTOR_DIM = len(test_vec)
    print(f"📏 Dimensão detectada: {REAL_VECTOR_DIM} (Ajustado automaticamente)")

    try:
        info = _redis_client.ft(INDEX_NAME).info()
    except:
        print(f"⚙️ Criando índice vetorial no Redis (DIM: {REAL_VECTOR_DIM})...")
        schema = (
            TextField("texto_pergunta"), 
            TextField("resposta"),       
            VectorField("vector",        
                "HNSW", {
                    "TYPE": "FLOAT32", 
                    "DIM": REAL_VECTOR_DIM, 
                    "DISTANCE_METRIC": "COSINE"
                }
            ),
        )
        definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
        _redis_client.ft(INDEX_NAME).create_index(schema, definition=definition)

except Exception as e:
    print(f"⚠️ REDIS STACK SETUP ERROR: {e}")
    if "Index already exists" not in str(e):
        print("💡 DICA: Se mudou o modelo de Embed, rode 'docker exec -it juridico_redis redis-cli FLUSHALL'")
    USE_REDIS = False


def buscar_com_cache_semantico(engine: BaseQueryEngine, pergunta_usuario: str) -> str:
    if not USE_REDIS:
        return str(engine.query(pergunta_usuario))

    try:
        try:
            info = _redis_client.ft(INDEX_NAME).info()
            num_docs = info.get(b'num_docs') or info.get('num_docs') or 0
            num_docs = int(num_docs)
        except:
            num_docs = 0
        
        vector = embed_model.get_query_embedding(pergunta_usuario)
        if len(vector) != REAL_VECTOR_DIM:
            print(f"⚠️ ALERTA: Vetor gerado ({len(vector)}) diferente do índice ({REAL_VECTOR_DIM})")
        
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()

        query = (
            Query("*=>[KNN 1 @vector $query_vector AS vector_score]")
            .sort_by("vector_score")
            .return_fields("vector_score", "resposta", "texto_pergunta")
            .dialect(2)
        )
        
        params = {"query_vector": vector_bytes}
        results = _redis_client.ft(INDEX_NAME).search(query, query_params=params)

        if results.docs:
            doc = results.docs[0]
            distancia = float(doc.vector_score)
            LIMIAR_ACEITAVEL = 0.35 

            if distancia < LIMIAR_ACEITAVEL:
                resposta_cache = doc.resposta
                if isinstance(resposta_cache, bytes):
                    resposta_cache = resposta_cache.decode('utf-8')
                print(f"⚡ CACHE HIT! (Dist: {distancia:.4f}) | Docs Indexados: {num_docs}")
                return resposta_cache
            else:
                print(f"💨 Cache Miss (Dist: {distancia:.4f}). Docs: {num_docs}")
        else:
            print(f"💨 Cache Miss (Zero vizinhos). Docs: {num_docs}")
        
        print(f"🔍 QDRANT: Processando pergunta inédita...")
        response = engine.query(pergunta_usuario)
        resposta_final = str(response)

        key = f"cache:{gerar_hash_estavel(pergunta_usuario)}"
        _redis_client.hset(key, mapping={
            b"vector": vector_bytes,
            b"texto_pergunta": pergunta_usuario.encode('utf-8'), 
            b"resposta": resposta_final.encode('utf-8')          
        })
        _redis_client.expire(key, 86400) 
        return resposta_final

    except Exception as e:
        print(f"⚠️ Erro no fluxo semântico: {e}")
        return str(engine.query(pergunta_usuario))

# =======================================================
# 2. CONFIGURAÇÃO DE DEPENDÊNCIAS
# =======================================================
@dataclass
class LegalDeps:
    query_engine: BaseQueryEngine
    historico_conversa: List[dict]
    documento_texto: str = ""

# =======================================================
# 3. TOOLS
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
# 4. AGENTES
# =======================================================
router_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps)

@router_agent.system_prompt
def prompt_router(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.router_tmpl.format(historico_conversa=ctx.deps.historico_conversa)

# --- Agente Simples ---
simples_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps, tools=[tool_buscar_rag, tool_pesquisa_web])
@simples_agent.system_prompt
def prompt_simples(ctx: RunContext[LegalDeps]) -> str:
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    # CORREÇÃO: "texto_documento" (nome no prompt) recebe "deps.documento_texto" (valor da dep)
    return Prompts.simples_tmpl.format(
        historico_conversa=ctx.deps.historico_conversa, 
        data_atual=data_hoje,
        texto_documento=ctx.deps.documento_texto 
    )

# --- Agente Corporativo ---
corporativo_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps, tools=[tool_buscar_rag, tool_pesquisa_web])
@corporativo_agent.system_prompt
def prompt_corporativo(ctx: RunContext[LegalDeps]) -> str:
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    # CORREÇÃO: "texto_documento"
    return Prompts.corporativo_tmpl.format(
        historico_conversa=ctx.deps.historico_conversa, 
        data_atual=data_hoje,
        texto_documento=ctx.deps.documento_texto
    )

# --- Agente Trabalhista ---
trabalhista_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps, tools=[tool_buscar_rag, tool_pesquisa_web])
@trabalhista_agent.system_prompt
def prompt_trabalhista(ctx: RunContext[LegalDeps]) -> str:
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    # CORREÇÃO: "texto_documento"
    return Prompts.trabalhista_tmpl.format(
        historico_conversa=ctx.deps.historico_conversa, 
        data_atual=data_hoje,
        texto_documento=ctx.deps.documento_texto
    )

# --- Agente Societário ---
societario_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps, tools=[tool_buscar_rag, tool_pesquisa_web])
@societario_agent.system_prompt
def prompt_societario(ctx: RunContext[LegalDeps]) -> str:
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    # CORREÇÃO: "texto_documento"
    return Prompts.societario_tmpl.format(
        historico_conversa=ctx.deps.historico_conversa, 
        data_atual=data_hoje,
        texto_documento=ctx.deps.documento_texto
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