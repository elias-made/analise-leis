from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# Imports do seu projeto
import LLM
import Langgraph
import ingestion  # O arquivo que acabamos de corrigir
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ==============================================================================
# 1. MODELOS DE DADOS (Pydantic)
# ==============================================================================

# --- Chat ---
class QueryRequest(BaseModel):
    data: str  # A pergunta do usuário

class Fonte(BaseModel):
    artigo: str
    lei: str
    link: str
    score: float

class QueryResponse(BaseModel):
    resposta: str
    fontes: List[Fonte]

# --- Ingestão ---
class IngestionRequest(BaseModel):
    urls: List[str]  # Recebe uma lista de URLs

class IngestionResponse(BaseModel):
    mensagem: str
    status: str

# ==============================================================================
# 2. CICLO DE VIDA (Lifespan)
# ==============================================================================
# Variável global para guardar a engine carregada na memória
query_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Inicializando API e carregando modelos...")
    global query_engine
    
    # 1. Configurações Globais do LlamaIndex
    Settings.embed_model = LLM.embed_model
    Settings.llm = LLM.llm_sonnet

    # 2. Conecta ao Qdrant (Modo Leitura Rápida)
    # Não precisamos configurar batch_size aqui, pois é só para leitura
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    vector_store = QdrantVectorStore(
        collection_name="leis_v2",
        client=client,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25"
    )
    
    # 3. Carrega o Índice (Instantâneo, pois lê do disco)
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        # 4. Cria a Engine de Busca
        query_engine = index.as_query_engine(
            similarity_top_k=5,       # 5 trechos por similaridade semântica
            sparse_top_k=10,          # 10 trechos por palavra-chave
            vector_store_query_mode="hybrid"
        )
        print("✅ Cérebro Jurídico carregado com sucesso!")
    except Exception as e:
        print(f"⚠️ Aviso: O banco pode estar vazio ou inacessível: {e}")
        print("A API vai iniciar, mas o chat pode falhar até que a ingestão seja feita.")
    
    yield
    
    print("🛑 Desligando API...")

# ==============================================================================
# 3. ROTAS DA API
# ==============================================================================
app = FastAPI(title="API Jurídica RAG", lifespan=lifespan)

@app.post("/perguntar")
async def ask_law(request: QueryRequest):
    # Cria o estado inicial com a pergunta do usuário
    estado_inicial = {"user_question": request.data}
    
    # Roda o grafo
    resultado_final = await Langgraph.ainvoke(estado_inicial)
    
    # O resultado_final é um dicionário com os campos do Dataclass preenchidos
    return {
        "resposta": resultado_final["final_response"],
        "perfil_detectado": resultado_final["classification_profile"]
    }


@app.post("/ingestion", response_model=IngestionResponse)
async def endpoint_ingestao(request: IngestionRequest, background_tasks: BackgroundTasks):
    """
    Endpoint de Ingestão.
    Recebe URLs -> Coloca na fila de processamento -> Retorna OK imediatamente.
    """
    if not request.urls:
        raise HTTPException(status_code=400, detail="A lista de URLs não pode estar vazia.")

    # AQUI É O PULO DO GATO:
    # Não fazemos loop, nem verificação de banco aqui.
    # Apenas passamos a lista crua para o ingestion.py se virar em background.
    background_tasks.add_task(ingestion.processar_urls_background, request.urls)

    return IngestionResponse(
        mensagem=f"Recebemos {len(request.urls)} URLs. O processamento iniciou em segundo plano.",
        status="processing_started"
    )