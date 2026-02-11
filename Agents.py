import os
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

# --- MUDANÇA CRUCIAL: Variável global para guardar a dimensão real ---
REAL_VECTOR_DIM = 1024 

USE_REDIS = False
_redis_client = None

def gerar_hash_estavel(texto: str) -> str:
    """Gera um ID único e constante para a mesma string (MD5)."""
    return hashlib.md5(texto.encode('utf-8')).hexdigest()

try:
    # 1. Conexão Binária
    redis_url = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/0")
    _redis_client = redis.Redis.from_url(redis_url, decode_responses=False) 
    _redis_client.ping()
    print(f"✅ REDIS STACK: Conectado (Modo Binário).")
    USE_REDIS = True

    # 2. AUTO-DETECÇÃO DE DIMENSÃO (O Segredo!)
    # Geramos um embedding de teste para ver o tamanho real que o modelo cospe.
    print("📏 Medindo dimensão do embedding...")
    test_vec = embed_model.get_query_embedding("teste de dimensao")
    REAL_VECTOR_DIM = len(test_vec)
    print(f"📏 Dimensão detectada: {REAL_VECTOR_DIM} (Ajustado automaticamente)")

    # 3. SETUP DO ÍNDICE
    try:
        info = _redis_client.ft(INDEX_NAME).info()
        # Se o índice já existe, precisamos ver se a dimensão bate.
        # Infelizmente o Redis não diz a dimensão fácil no info, então assumimos que se existe, tá ok.
        # Se der erro de dimensão depois, teria que apagar o índice (FLUSHALL).
    except:
        print(f"⚙️ Criando índice vetorial no Redis (DIM: {REAL_VECTOR_DIM})...")
        schema = (
            TextField("texto_pergunta"), 
            TextField("resposta"),       
            VectorField("vector",        
                "HNSW", {
                    "TYPE": "FLOAT32", 
                    "DIM": REAL_VECTOR_DIM, # Usa a dimensão real detectada
                    "DISTANCE_METRIC": "COSINE"
                }
            ),
        )
        definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
        _redis_client.ft(INDEX_NAME).create_index(schema, definition=definition)

except Exception as e:
    print(f"⚠️ REDIS STACK SETUP ERROR: {e}")
    # Se der erro de dimensão incompatível, avisa para limpar
    if "Index already exists" not in str(e):
        print("💡 DICA: Se mudou o modelo de Embed, rode 'docker exec -it juridico_redis redis-cli FLUSHALL'")
    USE_REDIS = False


def buscar_com_cache_semantico(engine: BaseQueryEngine, pergunta_usuario: str) -> str:
    if not USE_REDIS:
        return str(engine.query(pergunta_usuario))

    try:
        # Debug: Check docs count
        try:
            info = _redis_client.ft(INDEX_NAME).info()
            num_docs = info.get(b'num_docs') or info.get('num_docs') or 0
            num_docs = int(num_docs)
        except:
            num_docs = 0
        
        # 2. Gerar Embedding
        vector = embed_model.get_query_embedding(pergunta_usuario)
        
        # --- SEGURANÇA EXTRA: Verifica se a dimensão mudou no meio do caminho ---
        if len(vector) != REAL_VECTOR_DIM:
            print(f"⚠️ ALERTA: Vetor gerado ({len(vector)}) diferente do índice ({REAL_VECTOR_DIM})")
        
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()

        # 3. Buscar no Redis
        query = (
            Query("*=>[KNN 1 @vector $query_vector AS vector_score]")
            .sort_by("vector_score")
            .return_fields("vector_score", "resposta", "texto_pergunta")
            .dialect(2)
        )
        
        params = {"query_vector": vector_bytes}
        results = _redis_client.ft(INDEX_NAME).search(query, query_params=params)

        # 4. Analisar Resultados
        if results.docs:
            doc = results.docs[0]
            distancia = float(doc.vector_score)
            
            # LIMIAR
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
        
        # --- CACHE MISS: BUSCA NO QDRANT ---
        print(f"🔍 QDRANT: Processando pergunta inédita...")
        response = engine.query(pergunta_usuario)
        resposta_final = str(response)

        # 5. Salvar no Redis
        key = f"cache:{gerar_hash_estavel(pergunta_usuario)}"
        
        # Usamos chaves em bytes explicitamente para garantir compatibilidade no modo binário
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
    historico_conversa: str

# =======================================================
# 3. AGENTES
# =======================================================
router_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps)

@router_agent.system_prompt
def prompt_router(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.router_tmpl.format(historico_conversa=ctx.deps.historico_conversa)

# --- Agente Tributário ---
tributario_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps)
@tributario_agent.system_prompt
def prompt_tributario(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.tributario_tmpl.format(historico_conversa=ctx.deps.historico_conversa)
@tributario_agent.tool
def tool_buscar_tributario(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    return buscar_com_cache_semantico(ctx.deps.query_engine, termo_busca)

# --- Agente Trabalhista ---
trabalhista_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps)
@trabalhista_agent.system_prompt
def prompt_trabalhista(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.trabalhista_tmpl.format(historico_conversa=ctx.deps.historico_conversa)
@trabalhista_agent.tool
def tool_buscar_trabalhista(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    return buscar_com_cache_semantico(ctx.deps.query_engine, termo_busca)

# --- Agente Societário ---
societario_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps)
@societario_agent.system_prompt
def prompt_societario(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.societario_tmpl.format(historico_conversa=ctx.deps.historico_conversa)
@societario_agent.tool
def tool_buscar_societario(ctx: RunContext[LegalDeps], termo_busca: str) -> str:
    return buscar_com_cache_semantico(ctx.deps.query_engine, termo_busca)

# --- Agente Conversacional ---
conversational_agent = Agent(model=sonnet_bedrock_model, deps_type=LegalDeps)
@conversational_agent.system_prompt
def prompt_conversational(ctx: RunContext[LegalDeps]) -> str:
    return Prompts.conversational_tmpl.format(historico_conversa=ctx.deps.historico_conversa)

# Estrutura do veredito
class AvaliacaoJuiz(BaseModel):
    nota: int = Field(description="Nota de 1 a 5 para a resposta")
    justificativa: str = Field(description="Explicação do porquê da nota")
    tem_alucinacao: bool = Field(description="Se a IA inventou fatos fora dos documentos")
    correcao_necessaria: str = Field(description="O que o agente deve mudar se a nota for baixa")

# --- Agente Juiz ---
judge_agent = Agent(
    model=sonnet_bedrock_model, 
    deps_type=LegalDeps,
    output_type=AvaliacaoJuiz
)