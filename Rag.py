import os
import redis
import numpy as np
import hashlib

# Imports do Redis Stack
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from llama_index.core.base.base_query_engine import BaseQueryEngine

from LLM import (
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